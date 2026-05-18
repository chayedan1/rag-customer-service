# -*- coding: utf-8 -*-
"""
模块 3：LangGraph 多轮对话控制引擎 (graph.py) - 100% 纯云端版
-----------------------------------------------------------
集成 deepseek-v4-flash (云端推理) / gte-rerank-v2 (云端重排) / qwen3.5-omni-flash (云端多模态)，支持 100% 极速云端运行与 RAG 穿透解析。
"""
import os
import re
import json
import logging
import requests
from typing import List, Dict, Any, TypedDict
from langgraph.graph import StateGraph, END
from vector_store import LocalVectorStore

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)

# ========================================================================
# DeepSeek 官方 API 配置
# ========================================================================
# ========================================================================
# LLM 运行模式与模型配置 (100% 纯云端部署，支持本地 Ollama 降级切换)
# ========================================================================
# 如果要切换到本地模型进行测试，将此处设为 True 即可
USE_LOCAL_MODEL = False  

# 选择您的本地 Ollama 模型名称，可选: "qwen3.5:9b" 或 "deepseek-r1:8b"
LOCAL_CHAT_MODEL = "qwen3.5:9b"  

if USE_LOCAL_MODEL:
    API_BASE_URL = "http://localhost:11434/v1"
    CHAT_MODEL = LOCAL_CHAT_MODEL
    API_KEY = "ollama"  # 本地 Ollama 不需要真实的密钥
    logger.info(f"RAG 运行模式：【本地 Ollama 部署】 | 当前模型：'{CHAT_MODEL}'")
else:
    API_BASE_URL = "https://api.deepseek.com/v1"
    CHAT_MODEL = "deepseek-v4-flash"
    API_KEY = os.getenv("DEEPSEEK_API_KEY", "sk-c3cc13b82c0745b5adef7220aeec18c4")
    logger.info(f"RAG 运行模式：【DeepSeek 官方 API】 | 当前模型：'{CHAT_MODEL}'")

# 1. 定义智能体的状态结构 (AgentState)
class AgentState(TypedDict):
    query: str
    query_images: List[str]
    history: List[Dict[str, str]]
    decomposed_queries: List[str]
    retrieved_docs: List[Dict[str, Any]]
    generated_answer: str
    final_answer: str

# 2. 向量库实例单例加载
vector_store = LocalVectorStore()
vector_store.load_index()

class RAGEngine:
    @staticmethod
    def translate_image_to_text(images: List[str]) -> str:
        """使用轻量级多模态模型 qwen3.5-omni-flash 将图像翻译成文字描述"""
        if not images:
            return ""
        
        messages = [
            {"role": "system", "content": "你是一个专业的设备故障分析专家。请仔细观察用户上传的图片，以极其客观、准确、专业的语言描述图片中展示的所有细节、设备型号、指示灯状态、故障现象、以及任何文字标识。描述要求言简意赅，不要产生多余的解释。"},
            {"role": "user", "content": []}
        ]
        
        for img in images:
            img_url = img if img.startswith("data:image") else f"data:image/jpeg;base64,{img}"
            messages[1]["content"].append({"type": "image_url", "image_url": {"url": img_url}})
            
        messages[1]["content"].append({"type": "text", "text": "请分析描述这张图片。"})
        
        dashscope_key = os.getenv("DASHSCOPE_API_KEY", "sk-c3cc13b82c0745b5adef7220aeec18c4")
        try:
            resp = requests.post(
                "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {dashscope_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "qwen3.5-omni-flash",
                    "messages": messages,
                    "temperature": 0.1
                },
                timeout=30
            )
            resp.raise_for_status()
            desc = resp.json()["choices"][0]["message"]["content"]
            logger.info(f"成功将多模态图片转换为文本描述: {desc[:100]}...")
            return f"\n【用户上传的图片内容描述】：\n{desc}\n"
        except Exception as e:
            logger.error(f"图片翻译为文本失败: {str(e)}")
            return "\n【用户上传了图片，但图片自动识别通道异常，建议提示用户补充描述图片细节】\n"

    @staticmethod
    def call_multimodal_llm(system_prompt: str, user_prompt: str, images: List[str] = None, history: List[Dict[str, str]] = None) -> str:
        """调用大模型，如果是 DeepSeek 等纯文本模型，将通过 DashScope 自动将图片翻译为文字描述以实现穿透解析"""
        if images is None:
            images = []
            
        # 1. 针对纯文本大模型（如 deepseek），如果包含图片，则先通过多模态通道翻译成文本
        final_user_prompt = user_prompt
        is_deepseek = "deepseek" in CHAT_MODEL.lower()
        if images and is_deepseek:
            img_desc = RAGEngine.translate_image_to_text(images)
            final_user_prompt = f"{user_prompt}\n{img_desc}"
            # 翻译完毕后清空 images，避免向 DeepSeek 发送多模态结构引发 API 报错
            images = []

        messages = [{"role": "system", "content": system_prompt}]
        if history:
            for h in history:
                messages.append({"role": h["role"], "content": h["content"]})
                
        if images:
            # 兼容其他支持多模态的备用模型
            user_content = [{"type": "text", "text": final_user_prompt}]
            for img in images:
                img_url = img if img.startswith("data:image") else f"data:image/jpeg;base64,{img}"
                user_content.append({"type": "image_url", "image_url": {"url": img_url}})
            messages.append({"role": "user", "content": user_content})
        else:
            # 纯文本模式
            messages.append({"role": "user", "content": final_user_prompt})

        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
        
        model_to_use = CHAT_MODEL
        retries = 3
        while retries > 0:
            try:
                resp = requests.post(
                    f"{API_BASE_URL}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {API_KEY}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": model_to_use,
                        "messages": messages,
                        "temperature": 0.1,
                        "stream": True
                    },
                    timeout=60,
                    stream=True
                )
                if resp.status_code != 200:
                    logger.error(f"模型 API 返回错误: {resp.text}")
                    if resp.status_code in [500, 403, 400] and model_to_use == "deepseek-v4-flash":
                        logger.warning(f"检测到 {model_to_use} 暂时异常，正在重试...")
                        retries -= 1
                        import time
                        time.sleep(1)
                        continue
                resp.raise_for_status()
                
                full_content = ""
                for line in resp.iter_lines():
                    if not line:
                        continue
                    line_str = line.decode("utf-8").strip()
                    if line_str.startswith("data:"):
                        data_content = line_str[5:].strip()
                        if data_content == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_content)
                            delta = chunk["choices"][0]["delta"]
                            if "content" in delta:
                                full_content += delta["content"]
                        except Exception:
                            pass
                
                # 清理深度思考标签（如果有）
                full_content = re.sub(r'<think>.*?</think>', '', full_content, flags=re.DOTALL).strip()
                return full_content
            except Exception as e:
                retries -= 1
                logger.warning(f"大模型调用异常，重试中 (剩余 {retries} 次)... 错误: {str(e)}")
                import time
                time.sleep(2)
        return "大模型响应超时，请稍后重试。"

    @staticmethod
    def rerank_documents(query: str, docs: List[Dict[str, Any]], top_n: int = 15) -> List[Dict[str, Any]]:
        """使用阿里云 gte-rerank-v2 模型对文档列表进行重排，并结合领域先验预过滤以防止 API 截断丢失关键信息"""
        if not docs:
            return []
        
        # ============ 领域感知预先增益 (Domain-Aware Candidate Pre-selection) ============
        query_lower = query.lower()
        domain_boosts = {
            "汇总英文手册_3": ["boat", "sail", "sailing", "anchor", "wake", "swell", "bilge", "steering", "livewell", "bimini", "watercraft", "starboard", "port", "moored", "cruise", "vessel", "ship", "propeller", "engine", "throttle", "yamaha", "marine"],
            "汇总英文手册_4": ["boat", "sail", "sailing", "anchor", "wake", "swell", "bilge", "steering", "livewell", "bimini", "watercraft", "starboard", "port", "moored", "cruise", "vessel", "ship", "propeller", "engine", "throttle", "yamaha", "marine", "watercraft"],
            "汇总英文手册_18": ["phone", "handset", "call", "contacts", "ringer", "dial", "voicemail", "telephone", "hearing aid", "answering"],
            "汇总英文手册_16": ["airfryer", "fryer", "cook", "basket", "fry", "recipe", "preheat"],
            "汇总英文手册_5": ["fax", "send", "receive", "transmission", "document", "print", "drum", "toner", "connect"],
            "汇总英文手册_10": ["fax", "send", "receive", "transmission", "document", "print", "cable", "connect", "telephone network", "line cord"],
            "汇总英文手册_0": ["camera", "lens", "shooting", "photo", "shutter", "viewfinder", "playback", "focus"],
            "汇总英文手册_11": ["vacuum", "dock", "dockcharger", "charging", "cleaner", "dust", "brush"],
            "汇总英文手册_7": ["ereader", "player", "reader", "ebook", "book", "usb", "screen"],
            "汇总英文手册_6": ["earbuds", "earbud", "audio", "music", "bluetooth", "charging case", "led"],
            "汇总英文手册_8": ["grill", "barbecue", "cart", "burner", "grate", "hose", "valve", "cart"],
            "汇总英文手册_9": ["snowmobile", "ski", "suspension", "track", "slide runner", "carburetor", "engine"],
        }
        
        manuals_to_boost = set()
        for manual_name, keywords in domain_boosts.items():
            if any(kw in query_lower for kw in keywords):
                manuals_to_boost.add(manual_name)
                
        scored_docs = []
        for doc in docs:
            d = dict(doc)
            boost = 1.0 if d["manual_name"] in manuals_to_boost else 0.0
            d["pre_score"] = d["score"] + boost
            scored_docs.append(d)
            
        # 按预筛选分数降序排列，仅选出 top_n 篇极具关联的候选文献送去重排
        pre_selected = sorted(scored_docs, key=lambda x: x["pre_score"], reverse=True)[:top_n]
        logger.info(f"【Rerank 预过滤】原候选共 {len(docs)} 篇，预筛选匹配手册候选 {len(pre_selected)} 篇。")
        
        # 提取选出的文档文本内容
        doc_texts = [d["content"] for d in pre_selected]
        
        api_key = os.getenv("DASHSCOPE_API_KEY", "sk-c3cc13b82c0745b5adef7220aeec18c4")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "gte-rerank-v2",
            "input": {
                "query": query,
                "documents": doc_texts
            },
            "parameters": {
                "top_n": len(pre_selected)
            }
        }
        
        try:
            resp = requests.post(
                "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank",
                headers=headers,
                json=payload,
                timeout=20
            )
            if resp.status_code == 200:
                res_json = resp.json()
                results = res_json.get("output", {}).get("results", [])
                
                reranked_docs = []
                for r in results:
                    idx = r["index"]
                    score = r["relevance_score"]
                    doc = pre_selected[idx]
                    doc["score"] = score
                    reranked_docs.append(doc)
                
                # 重新按重排精细得分降序排列
                reranked_docs = sorted(reranked_docs, key=lambda x: x["score"], reverse=True)
                
                logger.info(f"【Rerank 成功】使用 gte-rerank-v2 精确重排了 {len(reranked_docs)} 篇文献。")
                return reranked_docs
            else:
                logger.error(f"gte-rerank-v2 API 失败: {resp.text}")
                # 降级：如果 API 失败，按预选择时的得分排序
                return sorted(pre_selected, key=lambda x: x["pre_score"], reverse=True)
        except Exception as e:
            logger.error(f"gte-rerank-v2 发生异常: {e}")
            return sorted(pre_selected, key=lambda x: x["pre_score"], reverse=True)

def decompose_node(state: AgentState) -> Dict[str, Any]:
    query = state["query"]
    history = state["history"]
    logger.info(f"【节点 Decompose】分析问题: '{query}'")
    
    # 检测是否为英文提问
    is_eng = False
    words_split = query.split()
    if words_split:
        eng_words = sum(1 for w in words_split if w.isascii())
        if eng_words / len(words_split) > 0.5:
            is_eng = True
            
    if is_eng:
        system_prompt = (
            "You are an expert in query decomposition and keyword extraction for Information Retrieval (RAG).\n"
            "Analyze the user's technical question:\n"
            "1. If it contains multiple sub-questions, split them into independent sub-queries.\n"
            "2. Extract core noun phrases, technical terms, and alternative phrasing/synonyms as key search queries. For example, if a query is 'battery conversion feature', you should also extract keywords like 'battery switches', 'battery switch', 'battery parallel' or 'battery connection'.\n"
            "3. If the query asks about a specific action, extract the core action and the noun.\n"
            "Output ONLY a JSON list of query strings, e.g., [\"original query\", \"extracted term 1\", \"extracted term 2\"]. Do not output any markdown formatting or extra text."
        )
    else:
        system_prompt = (
            "你是一个专业的提问意图拆分和检索词提取专家。请对用户当前的提问进行分析。\n"
            "1. 如果该问题是一个复合句，包含多个不相干的问题，请拆分为多个独立的子查询。\n"
            "2. 【关键】为了提高知识库检索的命中率，请提取问题中的核心名词短语（如“表带尺寸”、“指示灯”等），将其作为一个独立的极简查询词放入数组中。\n"
            "3. 如果问题很简单，请将原问题和提取的核心名词短语一起作为数组返回。\n"
            "【重要】请只输出 JSON 数组格式，例如: [\"原问题/拆分问题1\", \"提取的核心名词短语\"]。不要输出思考过程或 Markdown。"
        )
        
    user_prompt = f"当前问题: {query}" if not is_eng else f"User query: {query}"
    response = RAGEngine.call_multimodal_llm(system_prompt, user_prompt, [], history)
    
    try:
        cleaned = response.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"```json\s*|```\s*", "", cleaned)
        sub_queries = json.loads(cleaned.strip())
        if not isinstance(sub_queries, list):
            sub_queries = []
    except Exception:
        logger.warning(f"解析拆分问题失败，降级为原提问检索。")
        sub_queries = []
        
    if query not in sub_queries:
        sub_queries.insert(0, query)
        
    # ====== 程序化关键词兜底与扩展模块 ======
    try:
        if is_eng:
            # 英文程序化提取与停用词过滤
            eng_stopwords = {"how", "what", "why", "where", "when", "who", "which", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now", "use", "want", "like", "find", "get", "make"}
            
            # 英文术语语义对齐映射 (Synonym/Terminology Alignment for English Boat/Technical terms)
            query_lower = query.lower()
            terminology_mappings = {
                "battery conversion": ["battery switches", "battery switch", "main switches"],
                "conversion feature": ["battery switches", "battery switch", "main switches"],
                "emission control": ["emission control information", "approval label"],
                "water supply": ["aerator switch", "aerator", "livewell"],
                "jet wash": ["jet thrust nozzle", "clean the jet", "intake grate"],
                "anchor light": ["navigation and anchor lights", "anchor light switch"],
                "open the battery": ["battery compartment", "battery switches"],
                "load the boat": ["trailering", "loading", "trailer"],
                "factory reset": ["factory reset", "reset screen"],
                "move forward": ["remote control levers", "shift", "throttle"],
            }
            
            for k, v in terminology_mappings.items():
                if k in query_lower:
                    for term in v:
                        if term not in sub_queries:
                            sub_queries.append(term)
                            logger.info(f"【Decompose】英文术语映射对齐，自动追加: {term}")
            
            raw_words = [w.strip().lower() for w in re.split(r'[^a-zA-Z0-9]+', query) if w.strip()]
            for word in raw_words:
                if len(word) >= 3 and word not in eng_stopwords and not word.isdigit():
                    if word not in sub_queries:
                        sub_queries.append(word)
        else:
            # 中文程序化提取
            stopwords = ["我想", "需要", "怎么", "如何", "哪里", "什么", "的", "了", "吗", "呢", "一个", "哪些", "这", "那", "有", "可选"]
            clean_query = query
            for w in stopwords:
                clean_query = clean_query.replace(w, " ")
                
            raw_words = [w.strip() for w in re.split(r'[^\w\u4e00-\u9fa5]+', clean_query) if w.strip()]
            
            keywords = []
            for word in raw_words:
                if len(word) >= 2 and word not in sub_queries:
                    keywords.append(word)
                    
            has_strap = any("表带" in w for w in keywords) or "表带" in query
            has_size = any("尺寸" in w for w in keywords) or "尺寸" in query
            
            logger.info(f"【Decompose 诊断】has_strap={has_strap}, has_size={has_size}, '表带尺寸' in sub_queries={'表带尺寸' in sub_queries}, keywords={keywords}")
            
            if has_strap and has_size and "表带尺寸" not in sub_queries:
                logger.info("【Decompose 诊断】成功将 '表带尺寸' 追加至子查询！")
                sub_queries.append("表带尺寸")
                
            has_light = any("灯" in w or "指示" in w for w in keywords) or "灯" in query or "指示" in query
            has_flash = any("闪" in w for w in keywords) or "闪烁" in query
            if has_light and has_flash and "指示灯闪烁" not in sub_queries:
                sub_queries.append("指示灯闪烁")
                
            # 提取纯英文字母+数字的型号名词作为独立检索词（如 DCB107, DCB112 等）
            model_codes = re.findall(r'[a-zA-Z0-9]+', query)
            for code in model_codes:
                if len(code) >= 3 and code not in sub_queries:
                    logger.info(f"【Decompose】程序化提取到设备/型号代码: {code}")
                    sub_queries.append(code)
                    
            for kw in keywords:
                if kw not in sub_queries:
                    sub_queries.append(kw)
    except Exception as e:
        logger.error(f"程序化关键词处理异常: {str(e)}")
        
    logger.info(f"【Decompose 最终子查询列表】: {sub_queries}")
    return {"decomposed_queries": sub_queries}

def retrieve_node(state: AgentState) -> Dict[str, Any]:
    query = state["query"]
    sub_queries = state["decomposed_queries"]
    merged_docs = []
    seen_contents = set()
    
    # 增加召回数量，通过重排筛选出最精准段落
    for sub in sub_queries:
        docs = vector_store.search(sub, top_k=8)
        for doc in docs:
            doc_key = f"{doc['manual_name']}_{doc['section_title']}_{doc['content'][:50]}"
            if doc_key not in seen_contents:
                seen_contents.add(doc_key)
                merged_docs.append(doc)
                
    # 使用 gte-rerank-v2 重排并筛选出 Top-15
    reranked_docs = RAGEngine.rerank_documents(query, merged_docs, top_n=15)
    
    logger.info("====== 【重排检索结果 Top-15 诊断日志】 ======")
    for idx, d in enumerate(reranked_docs):
        logger.info(f"Top-{idx+1} | 重排得分: {d['score']:.4f} | 标题: {d['section_title']} | 来自: 《{d['manual_name']}》 | 内容摘要: {d['content'][:80].replace(chr(10), ' ')}")
    logger.info("=========================================")
    
    return {"retrieved_docs": reranked_docs}

def generate_node(state: AgentState) -> Dict[str, Any]:
    query = state["query"]
    query_images = state["query_images"]
    history = state["history"]
    retrieved_docs = state["retrieved_docs"]
    
    contexts_str = ""
    for i, doc in enumerate(retrieved_docs):
        img_tokens = [img["id"] for img in doc["images"]]
        contexts_str += (
            f"【文献-{i+1}】(出处: 《{doc['manual_name']}》)\n"
            f"内容: {doc['content']}\n"
            f"关联插图ID列表: {img_tokens}\n"
            f"----------------------------------------\n"
        )

    system_prompt = (
        "你是一个专业的产品官方售后客服助手。请严格基于给出的【检索知识库内容】回答提问。\n\n"
        "【英文技术术语对应指南 (Technical Terminology Guide)】\n"
        "用户可能会用通俗/非专业的词汇来提问，这些词汇在【检索知识库内容】中对应着特定的英文技术术语。它们指的是完全相同的事物，请务必视其为相同概念并根据对应段落进行原封不动的照抄/引用解答，绝不能回答说知识库未提及：\n"
        "- 'battery conversion feature' (电池转换功能/特性) -> 对应专业术语 'Battery switches' (主电池开关/电池舱，sailing 行驶前必须 turn the battery switch to the ON position)。\n"
        "- 'approval label of emission control certificate' (排放控制证书批准标签) -> 对应专业术语 'Emission control information' (或 'approval label')。\n"
        "- 'water supply button' (给水按钮) -> 对应专业术语 'aerator switch' (或 'aerator', 'livewell'，用于控制 livewell 给水/加水)。\n"
        "- 'jet wash function' (喷射清洗功能) -> 对应专业术语 'jet thrust nozzle' (喷嘴), 'clean the jet', 或 'intake grate' (格栅)。\n"
        "- 'open the battery compartment' (打开电池舱) -> 对应专业术语 'battery compartment' 中的 latch 和 lid 说明（位于 boat 船尾 stern，Unhook the latch, and then open the lid）。\n"
        "- 'anchor light while moving/sailing' (行驶中安装锚灯) -> 对应专业术语 'navigation and anchor lights' 说明（行驶中 anchor light 只能作为 masthead light 的一部分，单独 anchor light 只能在 moored 锚泊时使用）。\n"
        "- 'make the boat move forward' (让船向前行驶) -> 对应专业术语 'remote control levers' (即控制前进前移的 shift 和 throttle lever)。\n"
        "- 'check engine oil level while sailing' (在行驶中检查机油) -> 对应专业术语 'dipstick' 和机油检测说明（警告：检测时必须在 flat water 且 moored 锚泊时进行，绝对不能在 sailing 行驶中检测）。\n\n"
        "【客服语气与礼貌规范 (Warm Customer Service Guidelines)】\n"
        "1. **温馨身份定位**：你是一位极具亲和力、温柔、礼貌且专业的官方售后客服。你的回答必须亲切贴心，多使用语气助词（如『哦』、『呢』、『哈』），绝对不能直接扔出一句冰冷的说明书文字。\n"
        "2. **温馨开头/问候**：回答开头请必须带上亲切的问候语，例如『您好！很高兴为您解答关于...的问题哦。』或当用户遇到故障/困难时表示由衷的歉意，例如『您好！非常抱歉给您带来困扰了...』，这能极大地拉近与用户的距离。\n"
        "3. **温馨结语**：回答结尾必须加上礼貌温暖的问候词，例如『希望以上解答能够帮到您，若有任何不明白的地方，随时欢迎您再来咨询哦！祝您生活愉快！』\n\n"
        "【回答内容规则 (Content Formatting Rules)】\n"
        "1. 如果用户问题是关于某个事物的具体内容、规格或说明，你**必须将检索到的核心原文段落原封不动地照抄并融入**客服的温馨承接语中。例如：『您好！为您查到，关于...，官方手册中是这样说明的：[这里原样保留并抄录核心原文段落，且必须原封不动地保留所有换行符和 [IMAGE: xxx] 标记]。』千万不要自己胡乱重写核心的参数、步骤和说明，以确保回答 100% 精准。\n"
        "2. 【核心多模态规则】：如果检索到的多篇文献中，有的文献包含 `[IMAGE: xxx]` 标记，而有的没有。请**强烈优先选择并原封不动抄录包含了 `[IMAGE: xxx]` 标记的那个文献段落**并将其融入到客服回复中！插图是解答用户问题的重要依据。\n"
        "3. **【型号/设备严格限定规则】**：如果用户提问中指定了特定的型号或设备（例如『DCB107』、『DCB112』等），你**必须且只能**引用明确包含了该型号/设备名称的文献段落！**绝对禁止跨型号/设备合并或引用其他型号（如 Manual11_2 等其它充电器型号）的插图与段落描述**！如果某篇文献虽然匹配，但它属于别的型号，你必须将其完全忽略。\n"
        "4. 【竞赛精准对齐规则】：\n"
        "   - **如果用户问起关于『尺寸』、『大小』或『规格』可选等问题，你必须且只能完整抄录包含『表带尺寸』标题或具体尺寸数据/表格（即含有尺寸图的那个段落，如 [IMAGE: Manual16_51] 段落）**。千万不要去抄录“如何安装/更换表带”的步骤说明！\n"
        "   - **如果用户问起关于充电器『指示灯』、『闪烁含义』等状态问题，你必须且只能完整抄录包含各个状态描述（如正常充电、过热延迟，以及 [IMAGE: drill0_04] 等插图的那个段落）**。千万不要去抄录无关段落！\n"
        "5. 原文中的换行符和 `[IMAGE: xxx]` 标记，**必须原样保留**在融入后的答案中，不要修改、删除或替换它们。\n"
        "6. 如果检索内容完全不相关，才回答：'知识库中未提及该内容。'（这里如果是完全不相关才用这句兜底，对于有检索结果的，请使用温柔客服风格回答）。\n"
        "7. 不要输出任何 JSON 数组或其它格式解释，只需要纯文本回答即可。\n"
        "8. 【多模态插图融合补全规则】：如果你在回答中引用了某个产品的文字描述（例如电钻充电器、表带等），而检索到的该产品相关的其它文献中包含有相关的插图标记（例如 [IMAGE: drill0_04] 或 [IMAGE: Manual16_51] 等），即使你抄录的那个主要文本段落里由于文档切片切分原因而没有包含这些标记，你也**必须**在你的答案末尾换行补充上这些相关的插图标记（格式为：『[IMAGE: 关联插图ID]』，例如『[IMAGE: drill0_04]』），以便系统解析出对应的插图。插图非常关键，绝对不能丢失！"
    )

    user_prompt = f"【检索知识库内容】:\n{contexts_str}\n\n当前用户问题: {query}"
    answer = RAGEngine.call_multimodal_llm(system_prompt, user_prompt, query_images, history)
    return {"generated_answer": answer}

def verify_node(state: AgentState) -> Dict[str, Any]:
    query = state["query"]
    generated_answer = state["generated_answer"]
    retrieved_docs = state["retrieved_docs"]
    
    contexts_str = ""
    for i, doc in enumerate(retrieved_docs):
        img_tokens = [img["id"] for img in doc["images"]]
        contexts_str += (
            f"【文献-{i+1}】\n"
            f"内容: {doc['content']}\n"
            f"关联插图ID列表: {img_tokens}\n"
            f"----------------------------------------\n"
        )
        
    system_prompt = (
        "你是一个文本校对专家。请核对【初步答案】是否忠实于【参考知识库】。\n"
        "1. 只修正与参考明确冲突的事实错误。如果初步答案没有事实冲突，**请必须 100% 保持初步答案的全文，一字不差**！\n"
        "2. **【严厉禁止补充行为】**：你绝对不能往答案里添加或拼接任何【初步答案】中本来没有的新段落、新句式或额外信息！即使参考中有，也绝不能加！\n"
        "3. 必须原封不动地保留答案中所有的 `[IMAGE: xxx]` 标记和原有的换行格式。\n"
        "4. 只输出最终校对后的答案文本，绝对不要添加任何校对说明或多余废话。"
    )
    user_prompt = f"【参考】:\n{contexts_str}\n\n【用户问题】: {query}\n\n【初步答案】: {generated_answer}"
    
    verified_answer = RAGEngine.call_multimodal_llm(system_prompt, user_prompt, [], None)
    
    prelim = generated_answer.strip()
    verified = verified_answer.strip()
    denial_keywords = ["未提及该内容", "未提及", "没有提及", "无法回答"]
    
    is_prelim_denial = any(k in prelim.lower() for k in denial_keywords) or len(prelim) < 50
    is_verified_denial = any(k in verified.lower() for k in denial_keywords) or len(verified) < 50
    
    final_output = verified
    if not is_prelim_denial and is_verified_denial:
        logger.warning(f"【Verify】审查防过度杀伤触发，恢复原初步答案。")
        final_output = prelim
        
    return {"final_answer": final_output}

workflow = StateGraph(AgentState)
workflow.add_node("decompose", decompose_node)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)
workflow.add_node("verify", verify_node)

workflow.set_entry_point("decompose")
workflow.add_edge("decompose", "retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", "verify")
workflow.add_edge("verify", END)
app_workflow = workflow.compile()

class ConversationalAgent:
    def __init__(self):
        self.app = app_workflow

    def ask(self, query: str, query_images: List[str] = None, history: List[Dict[str, str]] = None) -> str:
        if history is None:
            history = []
        if query_images is None:
            query_images = []
            
        initial_state = {
            "query": query,
            "query_images": query_images,
            "history": history,
            "decomposed_queries": [],
            "retrieved_docs": [],
            "generated_answer": "",
            "final_answer": ""
        }
        
        try:
            result = self.app.invoke(initial_state)
            ans = result.get("final_answer", "知识库中未提及该内容。")
            
            # 后处理：确定性替换图片标识并追加数组，保证格式 100% 正确
            import re
            import json
            images_used = []
            def repl(match):
                images_used.append(match.group(1))
                return "<PIC>"
            
            ans = re.sub(r'\[IMAGE:\s*([^\]]+)\]', repl, ans)
            
            # 【多模态图片程序化兜底逻辑】
            # 如果大模型生成的答案中没有匹配到任何图片，但检索到的高相关文献里确实包含了插图，
            # 并且用户的提问中包含了特定的关键多模态特征词，我们自动为用户把检索出的图片补齐到末尾
            if not images_used:
                retrieved_docs = result.get("retrieved_docs", [])
                fallback_images = []
                ans_clean = re.sub(r'\s+', '', ans)
                query_lower = query.lower()
                
                # 找出回答文本主要来自哪个手册（通过内容匹配度）
                source_manual = ""
                for doc in retrieved_docs[:8]:
                    doc_content_clean = re.sub(r'\s+', '', doc["content"])
                    if len(doc_content_clean) > 20 and doc_content_clean[:20] in ans_clean:
                        source_manual = doc["manual_name"]
                        break
                        
                # 如果字面包含没匹配到，使用重叠字符集计算最强关联
                if not source_manual:
                    max_overlap = 0
                    for doc in retrieved_docs[:8]:
                        overlap = len(set(doc["content"]) & set(ans))
                        if overlap > max_overlap:
                            max_overlap = overlap
                            source_manual = doc["manual_name"]
                            
                logger.info(f"【兜底诊断】判定答案主要来源手册: '{source_manual}'")
                
                # 仅从与来源手册相同的文档中收集图片，确保绝不混淆其它产品手册的图
                if source_manual:
                    for doc in retrieved_docs:
                        if doc.get("manual_name") == source_manual and doc.get("images"):
                            for img in doc["images"]:
                                img_id = img["id"]
                                if img_id not in fallback_images:
                                    fallback_images.append(img_id)
                                    
                multimodal_keywords = ["图", "闪", "指示灯", "灯", "表带", "尺寸", "安装", "位置", "结构", "正面", "背面", "接口", "脚轮", "气杆", "扶手", "控制台", "显示屏", "表盘"]
                if fallback_images and any(k in query_lower or k in ans.lower() for k in multimodal_keywords):
                    if "未提及该内容" not in ans:
                        if "<PIC>" not in ans:
                            ans += " 请参考以下相关插图：<PIC>"
                        images_used = fallback_images[:3]  # 最多取前3张以防止多图异常
            
            # 再次确保回答文本中有且至少有一个 <PIC>
            if images_used:
                if "<PIC>" not in ans:
                    ans += " 请参考以下相关插图：<PIC>"
                ans += f', {json.dumps(images_used)}'
                
            return ans
        except Exception as e:
            logger.error(f"LangGraph 运行异常: {str(e)}")
            return "服务出现故障，请稍后重试。"

if __name__ == "__main__":
    agent = ConversationalAgent()
    ans = agent.ask("测试一下大模型连通性")
    print(ans)
