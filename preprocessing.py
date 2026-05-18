# -*- coding: utf-8 -*-
"""
模块 1：知识库预处理模块 (preprocessing.py)
----------------------------------------
功能说明：
1. 鲁棒性支持：自动检测 KnowledgeBase 目录，若未解压则自动从 KownledgeBase.zip 解压。
2. 遍历所有产品手册（包括中文手册和汇总英文手册）。
3. 严格解析产品手册的 JSON 格式数据 [text_content, image_list]。
4. 实现精准的图文绑定：将文本中的 <PIC> 占位符替换为具体的图片 ID [IMAGE: image_name]，并在分块中自动绑定。
5. 结构化分块：支持按 Markdown 标题层级和字数进行混合切片，保证上下文完整性。
6. 输出结构化的 JSON 缓存，以便后续向量库和 RAG 调用。
"""

import os
import re
import json
import zipfile
import logging
from typing import List, Dict, Any, Tuple

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)

class KBPreprocessor:
    def __init__(self, workspace_dir: str = "d:\\Desktop\\数据"):
        """
        初始化预处理器，设置工作区和知识库路径。
        """
        self.workspace_dir = workspace_dir
        self.kb_dir = os.path.join(workspace_dir, "KownledgeBase")
        self.manuals_dir = os.path.join(self.kb_dir, "手册")
        self.images_dir = os.path.join(self.manuals_dir, "插图")
        self.zip_path = os.path.join(workspace_dir, "KownledgeBase.zip")
        self.output_json_path = os.path.join(workspace_dir, "knowledge_processed.json")
        
        # 缓存图片文件Stem到绝对路径的映射
        self.image_map = {}

    def ensure_kb_extracted(self):
        """
        确保知识库压缩包已正确解压，若未解压则自动在本地解压。
        """
        if not os.path.exists(self.kb_dir):
            if os.path.exists(self.zip_path):
                logger.info(f"检测到 KnowledgeBase 目录不存在，正在从压缩包 {self.zip_path} 解压...")
                with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
                    # 解决解压时中文乱码问题
                    for member in zip_ref.infolist():
                        try:
                            # 尝试对文件名进行编码转换（通常 zip 文件在 Windows 上使用 cp437 或 gbk 编码）
                            filename = member.filename.encode('cp437').decode('gbk')
                        except Exception:
                            filename = member.filename
                        
                        target_path = os.path.join(self.workspace_dir, filename)
                        # 创建父目录
                        os.makedirs(os.path.dirname(target_path), exist_ok=True)
                        if not member.is_dir():
                            with zip_ref.open(member) as source, open(target_path, "wb") as target:
                                target.write(source.read())
                logger.info("解压完成！")
            else:
                raise FileNotFoundError(f"未找到 KownledgeBase.zip 且 {self.kb_dir} 目录不存在！")
        else:
            logger.info("检测到 KnowledgeBase 目录已存在，跳过解压。")
            
        # 检查是否成功解压并校验关键目录
        if not os.path.exists(self.manuals_dir):
            # 有时可能多了一层包装目录，尝试兼容性扫描
            logger.warning(f"目标手册目录 {self.manuals_dir} 不存在，正在进行递归路径搜寻...")
            for root, dirs, files in os.walk(self.kb_dir):
                if "手册" in dirs:
                    self.manuals_dir = os.path.join(root, "手册")
                    self.images_dir = os.path.join(self.manuals_dir, "插图")
                    logger.info(f"重新定位手册目录至: {self.manuals_dir}")
                    break

    def build_image_map(self):
        """
        遍历插图目录，构建图片 ID（无后缀文件名）与文件绝对路径的映射字典。
        """
        self.image_map = {}
        if not os.path.exists(self.images_dir):
            logger.warning(f"插图目录 {self.images_dir} 不存在！将无法绑定具体图片绝对路径。")
            return

        for root, _, files in os.walk(self.images_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                    stem = os.path.splitext(file)[0]
                    # 保存图片绝对路径
                    self.image_map[stem] = os.path.abspath(os.path.join(root, file))
        
        logger.info(f"图片映射构建完毕，共找到 {len(self.image_map)} 张插图文件。")

    def clean_json_string(self, raw_str: str) -> str:
        """
        清理 JSON 字符串中的非法转义符。
        使用负向断言过滤合法的双反斜杠以及合法的 JSON 转义，确保绝对不会对合法部分进行二次破坏。
        """
        # (?<!\\) 确保反斜杠前面没有另一个反斜杠（避免匹配已转义的双反斜杠 \\）
        # \\ 匹配单反斜杠本身
        # (?!["\\/bfnrtu]) 确保后面跟的不是合法 JSON 转义字符
        cleaned = re.sub(r'(?<!\\)\\(?!["\\/bfnrtu])', r'\\\\', raw_str)
        # 处理非法 \u 转义（当 \u 后面未跟 4 位十六进制数字时，将其进行双重转义转为字面反斜杠）
        cleaned = re.sub(r'(?<!\\)\\u(?![0-9a-fA-F]{4})', r'\\\\u', cleaned)
        return cleaned

    def parse_manual_data(self, content_str: str) -> List[Tuple[str, List[str]]]:
        """
        解析手册内容，返回列表：[(text_content, image_list), ...]
        兼容处理：
        1. 整个文件为一个大 JSON array。
        2. 文件包含多行，每行为一个完整的 JSON array（如汇总英文手册.txt）。
        """
        content_str = content_str.strip()
        parsed_data = []

        # 尝试整体解析
        try:
            cleaned_str = self.clean_json_string(content_str)
            data = json.loads(cleaned_str)
            if isinstance(data, list) and len(data) >= 2 and isinstance(data[0], str):
                parsed_data.append((data[0], data[1]))
                return parsed_data
        except Exception as e:
            # 整体解析失败，或者不是标准格式，尝试按行解析
            logger.debug(f"整体解析失败，转向逐行尝试解析: {str(e)}")

        # 逐行解析
        lines = content_str.split("\n")
        for line_num, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            try:
                cleaned_line = self.clean_json_string(line)
                data = json.loads(cleaned_line)
                if isinstance(data, list) and len(data) >= 2 and isinstance(data[0], str):
                    parsed_data.append((data[0], data[1]))
                else:
                    logger.warning(f"行 {line_num} 的数据结构不是标准 [text, image_list] 格式，跳过。")
            except Exception as le:
                logger.error(f"解析行 {line_num} 失败: {str(le)}")
                
        return parsed_data

    def process_single_manual(self, file_path: str) -> List[Dict[str, Any]]:
        """
        解析单本手册，实现分块与图文精准绑定。
        """
        manual_name = os.path.splitext(os.path.basename(file_path))[0]
        logger.info(f"开始解析手册: {manual_name}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content_str = f.read()
            
            parsed_data = self.parse_manual_data(content_str)
            if not parsed_data:
                logger.error(f"手册解析后无有效数据，文件名: {file_path}")
                return []
        except Exception as e:
            logger.error(f"读取手册 {file_path} 失败: {str(e)}")
            return []

        chunks = []
        MAX_CHUNK_LEN = 800  # 每个块的最大字符长度
        OVERLAP_LEN = 150    # 长句切分时的重叠字数

        def save_chunk(text_lines: List[str], section_title: str, sub_manual_idx: int):
            """内部辅助函数：生成并保存切片"""
            text_str = "\n".join(text_lines).strip()
            if not text_str:
                return
            
            # 匹配并抽取当前 Chunk 中包含的所有 [IMAGE: img_id] 图片
            chunk_images = re.findall(r"\[IMAGE:\s*([^\]]+)\]", text_str)
            image_details = []
            for img_id in chunk_images:
                # 查找图片绝对路径
                abs_path = self.image_map.get(img_id, "")
                image_details.append({
                    "id": img_id,
                    "path": abs_path
                })
                
            # 兼容多手册命名，避免冲突
            name_suffix = f"_{sub_manual_idx}" if len(parsed_data) > 1 else ""
            chunks.append({
                "manual_name": f"{manual_name}{name_suffix}",
                "section_title": section_title,
                "content": text_str,
                "images": image_details
            })

        for idx, (text_content, image_list) in enumerate(parsed_data):
            # 兼容性处理 1：有些手册文件（如吹风机手册、空气净化器手册）里的换行是以字符字面量 "\\n" 形式保存的，
            # 需要替换为真实的换行符以确保后续 split("\n") 能够正确分行。
            text_content = text_content.replace("\\n", "\n")
            
            # 兼容性处理 2：有些手册在打包打包时被扁平化成了单行文本，章节标题仅靠空格和井号 (如 " # 标题") 隔开，
            # 我们使用高容错正则检测并将其还原为真正的换行标题 "\n# 标题"，从而让分块器能完美提取各章节信息，避免整本手册丢失。
            text_content = re.sub(r" +(#{1,6} )", r"\n\1", text_content)
            
            # 1. 替换文本中的 <PIC> 为带有图片名称 of 显式 Token
            pic_pattern = re.compile(r"<PIC>")
            matches = list(pic_pattern.finditer(text_content))
            
            # 逐个替换 <PIC> 标签
            new_text = ""
            last_idx = 0
            for i, match in enumerate(matches):
                start, end = match.span()
                new_text += text_content[last_idx:start]
                if i < len(image_list):
                    img_id = image_list[i]
                    # 绑定成易于切片 and 提取的 Token 结构
                    new_text += f" [IMAGE: {img_id}] "
                else:
                    new_text += " " # 标签多于图片列表，用空格替代
                last_idx = end
            new_text += text_content[last_idx:]
            
            # 处理可能有多余图片但文本中未声明 <PIC> 的情况，将其附在末尾
            if len(image_list) > len(matches):
                extra_imgs = image_list[len(matches):]
                new_text += "\n\n# 附录插图\n"
                for img_id in extra_imgs:
                    new_text += f"[IMAGE: {img_id}]\n"

            # 2. 结构化切片与分块
            # 采用按标题行切分的策略，保持每个块的语意聚焦度，并在长段落时进行字数切分
            lines = new_text.split("\n")
            current_chunk_lines = []
            current_title = "前言/导言"
            current_len = 0
            
            for line in lines:
                line_strip = line.strip()
                # 检测 Markdown 标题
                if line_strip.startswith("#"):
                    # 如果当前有积压的内容，先保存
                    if current_chunk_lines:
                        save_chunk(current_chunk_lines, current_title, idx)
                        current_chunk_lines = []
                        current_len = 0
                    current_title = line_strip.lstrip("#").strip()
                    current_chunk_lines.append(line)
                    current_len += len(line)
                else:
                    current_chunk_lines.append(line)
                    current_len += len(line)
                    
                    # 如果当前块字数超限，进行截断切分，并保留重叠行
                    if current_len >= MAX_CHUNK_LEN:
                        save_chunk(current_chunk_lines, current_title, idx)
                        # 保留最后几行作为重叠部分
                        overlap_lines = []
                        overlap_len = 0
                        for rev_line in reversed(current_chunk_lines):
                            if overlap_len + len(rev_line) < OVERLAP_LEN:
                                overlap_lines.insert(0, rev_line)
                                overlap_len += len(rev_line)
                            else:
                                break
                        current_chunk_lines = overlap_lines
                        current_len = overlap_len
                        
            # 保存最后一个块
            if current_chunk_lines:
                save_chunk(current_chunk_lines, current_title, idx)

        logger.info(f"手册 {manual_name} 解析完成，共生成 {len(chunks)} 个文本分块。")
        return chunks

    def preprocess_all(self):
        """
        遍历并预处理工作区下 KnowledgeBase 中的所有文档。
        """
        self.ensure_kb_extracted()
        self.build_image_map()
        
        all_chunks = []
        
        # 扫描手册目录下所有 .txt 文件
        if not os.path.exists(self.manuals_dir):
            logger.error(f"手册目录不存在: {self.manuals_dir}")
            return
            
        for file in os.listdir(self.manuals_dir):
            if file.endswith(".txt"):
                file_path = os.path.join(self.manuals_dir, file)
                # 排除可能包含的非标准文件或临时文件
                if file.startswith("."):
                    continue
                chunks = self.process_single_manual(file_path)
                all_chunks.extend(chunks)
                
        # 写入全局缓存 JSON
        with open(self.output_json_path, 'w', encoding='utf-8') as f:
            json.dump(all_chunks, f, ensure_ascii=False, indent=2)
            
        logger.info(f"所有知识库文档预处理完毕！共产生 {len(all_chunks)} 个切片，结构化数据已写入 {self.output_json_path}")
        return all_chunks

if __name__ == "__main__":
    preprocessor = KBPreprocessor()
    preprocessor.preprocess_all()
