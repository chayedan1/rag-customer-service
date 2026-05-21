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

def is_english_query(query: str) -> bool:
    """Detect if the query is in English based on word/character ratio."""
    zh_chars = len(re.findall(r'[\u4e00-\u9fff]', query))
    en_words = len(re.findall(r'[a-zA-Z]{2,}', query))
    if zh_chars == 0 and en_words > 0:
        return True
    return en_words > zh_chars


def classify_zh_generic(query: str) -> bool:
    """
    \u5224\u65ad\u4e2d\u6587\u95ee\u9898\u662f\u5426\u4e3a\u901a\u7528\u5ba2\u670d\u95ee\u9898\uff08\u65e0\u9700 RAG\uff0c\u7528\u6a21\u677f\u56de\u7b54\uff09\u3002
    \u8fd4\u56de True \u8868\u793a\u662f\u901a\u7528\u5ba2\u670d\u95ee\u9898\uff0cFalse \u8868\u793a\u662f\u4ea7\u54c1\u64cd\u4f5c\u95ee\u9898\u5e94\u8d70 RAG\u3002
    """
    # \u5982\u679c\u95ee\u9898\u4e2d\u5305\u542b\u4ea7\u54c1\u540d\u79f0\uff0c\u5373\u4f7f\u6709\u5ba2\u670d\u5173\u952e\u8bcd\u4e5f\u5f52\u4e3a\u4ea7\u54c1\u95ee\u9898
    # Compound question detection: multi-clause questions should go to RAG
    # Enhanced compound detection: newlines, multiple periods, or mixed complaint topics
    _compound_indicators = ['质量问题', '少发', '少件', '发票开错', '翻新机', '假货',
                            '包装破损', '过期', '受潮', '维修天', '异味']
    _issue_count = sum(1 for ind in _compound_indicators if ind in query)
    if chr(10) in query or query.count('。') >= 2 or _issue_count >= 2:
        return False
        return False
        return False

    product_keywords = [
        '\u7a7a\u8c03', '\u51b0\u7bb1', '\u70e4\u7bb1', '\u9f20\u6807', '\u952e\u76d8', '\u7535\u94bb', '\u6d17\u7897\u673a', '\u5439\u98ce\u673a',
        'VR', '\u5934\u663e', '\u5065\u8eab\u5355\u8f66', '\u5065\u8eab\u8ffd\u8e2a', '\u6469\u6258\u8f66', '\u76f8\u673a', '\u7a7a\u6c14\u51c0\u5316\u5668', '\u6c34\u6cf5', '\u53d1\u7535\u673a',
        '\u6e29\u63a7\u5668', '\u84b8\u6c7d\u6e05\u6d01', '\u4eba\u4f53\u5de5\u5b66\u6905', '\u6469\u6258\u8247', '\u8ffd\u8e2a\u5668',
        'DCB', 'WIDCOMM', 'CAM\u8f6f\u4ef6',
        '\u8033\u673a', '\u8033\u585e', '\u906e\u5149\u7f69', '\u6e38\u73a9\u533a\u57df',
        '\u6d17\u6da4\u5757', '\u4eae\u789f\u5242', '\u9910\u5177\u7bee', '\u7897\u7bee', '\u55b7\u6dcb\u81c2',
        '\u6ee4\u7f51', '\u811a\u8f6e', '\u7070\u5c18\u4f20\u611f\u5668',
        '\u5939\u5934', '\u7535\u6c60\u7ec4', '\u8170\u5e26\u6302\u94a9', '\u6279\u5934\u5939',
        '\u8868\u5e26', '\u6263\u7d27\u8868\u5e26', '\u62c6\u5378\u8868\u5e26', '\u9501\u5c4f',
        '\u63a7\u5236\u53f0', '\u9884\u8bbe\u8fd0\u52a8', '\u5fc3\u7387\u76ee\u6807', '\u4f53\u80fd\u6d4b\u8bd5',
        '\u9065\u63a7\u5668', '\u767e\u53f6\u7a97', '\u5bfc\u98ce\u677f', '\u81ea\u6e05\u6d01', '\u7b49\u79bb\u5b50',
        '\u5236\u51b7', '\u5236\u70ed', '\u9664\u6e7f', '\u6362\u6c14', '\u98ce\u901f',
        '\u6e38\u6cf3\u5e73\u53f0', '\u6d3b\u9c7c\u8231', 'bilge', 'throttle',
        '\u6321\u6ce5\u677f', '\u524d\u8f6e', '\u9a71\u52a8\u7a0b\u5e8f', '\u914d\u5bf9', '\u5feb\u901f\u914d\u5bf9',
        '\u6d17\u8863', '\u6d17\u7897'
    ]
    for pk in product_keywords:
        if pk in query:
            return False

    # \u901a\u7528\u5ba2\u670d\u5173\u952e\u8bcd
    generic_keywords = [
        '7\u5929\u65e0\u7406\u7531', '\u4e03\u5929\u65e0\u7406\u7531', '\u9000\u6362\u8d27', '\u9000\u8d27', '\u9000\u6b3e',
        '\u53d1\u7968', '\u8fd0\u8d39', '\u5feb\u9012', '\u6295\u8bc9', '\u8865\u53d1', '\u7f3a\u4ef6', '\u5c11\u53d1',
        '\u4f18\u60e0\u5238', '\u4ee5\u65e7\u6362\u65b0', '\u4fdd\u8d28\u671f', '\u4e0a\u95e8\u5b89\u88c5', '\u5b89\u88c5\u670d\u52a1',
        '\u8bd5\u7528\u88c5', '\u7eb8\u8d28\u7248\u8bf4\u660e\u4e66', '\u7ec8\u8eab\u7ef4\u4fee', '\u667a\u80fd\u5ba2\u670d\u80fd\u89e3\u7b54',
        '\u5047\u8d27', '\u4e8c\u624b\u5546\u54c1', '\u7ffb\u65b0\u673a', '\u865a\u5047\u5ba3\u4f20', '\u989c\u8272\u504f\u5dee',
        '\u5305\u88c5\u7834\u635f', '\u5305\u88c5\u76d2\u4e22\u4e86', '\u53d6\u6d88\u8ba2\u5355', '\u552e\u540e\u4fdd\u969c\u5361',
        '\u5bc4\u5230\u56fd\u5916', '\u8d85\u65f6', '\u8d54\u507f', '\u5f00\u53d1\u7968',
        '\u552e\u5047', '\u5047\u5192', '\u7ffb\u65b0', '\u4e34\u671f', '\u8fc7\u671f',
        '\u552e\u540e\u7ef4\u4fee\u670d\u52a1', '\u552e\u540e\u5417', '\u8fd8\u80fd\u552e\u540e', '\u552e\u540e\u592a\u5dee',
        '\u5546\u54c1\u4fdd\u8d28\u671f', '\u5546\u54c1\u8d28\u91cf', '\u4f7f\u7528\u4e00\u6b21\u5c31\u574f\u4e86',
        '\u6362\u8d27\u5417', '\u80fd\u6362\u5417', '\u80fd\u6362\u8d27', '\u6211\u60f3\u6362\u8d27', '\u60f3\u6362\u8d27',
        '\u989c\u8272\u548c', '\u5f02\u5473', '\u5546\u54c1\u6709\u7455\u75b5',
    ]
    for gk in generic_keywords:
        if gk in query:
            return True

    return False


def get_zh_template_response(query: str) -> str:
    """\u4e3a\u4e2d\u6587\u901a\u7528\u5ba2\u670d\u95ee\u9898\u63d0\u4f9b\u4e13\u4e1a\u6a21\u677f\u56de\u7b54\u3002"""
    q = query

    # \u9000\u6362\u8d27\u76f8\u5173
    if any(kw in q for kw in ['7\u5929\u65e0\u7406\u7531', '\u4e03\u5929\u65e0\u7406\u7531', '\u9000\u6362\u8d27', '\u9000\u8d27', '\u6362\u8d27']):
        resp = (
            "\u60a8\u597d\uff01\u5f88\u9ad8\u5174\u4e3a\u60a8\u89e3\u7b54\u5173\u4e8e\u9000\u6362\u8d27\u7684\u95ee\u9898\u54e6\u3002\n\n"
            "\u5173\u4e8e\u9000\u6362\u8d27\u653f\u7b56\uff0c\u6211\u4eec\u7684\u6807\u51c6\u670d\u52a1\u6761\u6b3e\u5982\u4e0b\uff1a\n"
            "1. \u81ea\u7b7e\u6536\u5546\u54c1\u4e4b\u65e5\u8d777\u5929\u5185\uff0c\u652f\u6301\u65e0\u7406\u7531\u9000\u6362\u8d27\uff0c\u5546\u54c1\u9700\u4fdd\u6301\u5b8c\u597d\u3001\u672a\u7ecf\u4f7f\u7528\uff0c\u4e14\u5305\u88c5\u9644\u4ef6\u9f50\u5168\u3002\n"
            "2. 7\u5929\u5185\u9000\u6362\u8d27\u7684\u8fd0\u8d39\u627f\u62c5\uff1a\u5982\u56e0\u5546\u54c1\u8d28\u91cf\u95ee\u9898\u5bfc\u81f4\u7684\u9000\u6362\u8d27\uff0c\u8fd0\u8d39\u7531\u6211\u4eec\u627f\u62c5\uff1b\u5982\u56e0\u4e2a\u4eba\u539f\u56e0\uff08\u5982\u4e0d\u559c\u6b22\u3001\u4e70\u9519\u7b49\uff09\uff0c\u8fd0\u8d39\u9700\u7531\u4e70\u5bb6\u81ea\u884c\u627f\u62c5\u3002\n"
            "3. \u8d85\u8fc77\u5929\u4f46\u5728\u4fdd\u4fee\u671f\u5185\uff0c\u5982\u5546\u54c1\u51fa\u73b0\u8d28\u91cf\u95ee\u9898\uff0c\u53ef\u901a\u8fc7\u552e\u540e\u7ef4\u4fee\u6e20\u9053\u5904\u7406\u3002\n"
            "4. \u6362\u8d27\u65f6\u8bf7\u786e\u4fdd\u5546\u54c1\u539f\u59cb\u5305\u88c5\u5b8c\u597d\uff0c\u5982\u5305\u88c5\u4e22\u5931\u53ef\u80fd\u4f1a\u5f71\u54cd\u6362\u8d27\u6d41\u7a0b\uff0c\u5efa\u8bae\u60a8\u63d0\u524d\u4e0e\u5ba2\u670d\u786e\u8ba4\u3002\n\n"
            "\u5177\u4f53\u7684\u9000\u6362\u8d27\u64cd\u4f5c\uff0c\u5efa\u8bae\u60a8\u767b\u5f55\u6211\u4eec\u7684\u552e\u540e\u670d\u52a1\u5e73\u53f0\u63d0\u4ea4\u7533\u8bf7\uff0c\u6216\u76f4\u63a5\u8054\u7cfb\u4eba\u5de5\u5ba2\u670d\u4e3a\u60a8\u529e\u7406\u54e6\uff01\n\n"
            "\u5e0c\u671b\u4ee5\u4e0a\u89e3\u7b54\u80fd\u591f\u5e2e\u5230\u60a8\uff0c\u82e5\u6709\u4efb\u4f55\u4e0d\u660e\u767d\u7684\u5730\u65b9\uff0c\u968f\u65f6\u6b22\u8fce\u60a8\u518d\u6765\u54a8\u8be2\u54e6\uff01\u795d\u60a8\u751f\u6d3b\u6109\u5feb\uff01"
        )
        return resp

    # \u9000\u6b3e\u76f8\u5173
    if any(kw in q for kw in ['\u9000\u6b3e', '\u9000\u6b3e\u591a\u4e45', '\u5168\u989d\u9000\u6b3e']):
        resp = (
            "\u60a8\u597d\uff01\u5f88\u9ad8\u5174\u4e3a\u60a8\u89e3\u7b54\u5173\u4e8e\u9000\u6b3e\u7684\u95ee\u9898\u54e6\u3002\n\n"
            "\u5173\u4e8e\u9000\u6b3e\u653f\u7b56\uff0c\u4ee5\u4e0b\u4fe1\u606f\u4f9b\u60a8\u53c2\u8003\uff1a\n"
            "1. \u9000\u6b3e\u7533\u8bf7\u5ba1\u6838\u901a\u8fc7\u540e\uff0c\u4e00\u822c\u4f1a\u57283-7\u4e2a\u5de5\u4f5c\u65e5\u5185\u539f\u8def\u9000\u56de\u81f3\u60a8\u7684\u652f\u4ed8\u8d26\u6237\u3002\n"
            "2. \u4fe1\u7528\u5361\u652f\u4ed8\u7684\u9000\u6b3e\u5c06\u539f\u8def\u8fd4\u56de\u81f3\u60a8\u7684\u4fe1\u7528\u5361\u8d26\u6237\uff0c\u5230\u8d26\u65f6\u95f4\u53ef\u80fd\u56e0\u94f6\u884c\u5904\u7406\u901f\u5ea6\u7565\u6709\u5dee\u5f02\u3002\n"
            "3. \u5982\u8ba2\u5355\u5df2\u4ed8\u6b3e\u4f46\u5c1a\u672a\u53d1\u8d27\uff0c\u53d6\u6d88\u8ba2\u5355\u540e\u53ef\u7533\u8bf7\u5168\u989d\u9000\u6b3e\u3002\n"
            "4. \u5982\u5546\u54c1\u5df2\u7b7e\u6536\u540e\u7533\u8bf7\u9000\u6b3e\uff0c\u9700\u5148\u5b8c\u6210\u9000\u8d27\u6d41\u7a0b\uff08\u5546\u54c1\u5bc4\u56de\u5e76\u9a8c\u6536\u5408\u683c\uff09\u540e\u65b9\u53ef\u9000\u6b3e\u3002\n\n"
            "\u5982\u9700\u529e\u7406\u9000\u6b3e\uff0c\u5efa\u8bae\u60a8\u901a\u8fc7\u6211\u4eec\u7684\u552e\u540e\u670d\u52a1\u5e73\u53f0\u63d0\u4ea4\u9000\u6b3e\u7533\u8bf7\uff0c\u5ba2\u670d\u4f1a\u5c3d\u5feb\u4e3a\u60a8\u5904\u7406\u54e6\uff01\n\n"
            "\u5e0c\u671b\u4ee5\u4e0a\u89e3\u7b54\u80fd\u591f\u5e2e\u5230\u60a8\uff0c\u82e5\u6709\u4efb\u4f55\u4e0d\u660e\u767d\u7684\u5730\u65b9\uff0c\u968f\u65f6\u6b22\u8fce\u60a8\u518d\u6765\u54a8\u8be2\u54e6\uff01\u795d\u60a8\u751f\u6d3b\u6109\u5feb\uff01"
        )
        return resp

    # \u53d1\u7968\u76f8\u5173
    if any(kw in q for kw in ['\u53d1\u7968', '\u5f00\u7968']):
        resp = (
            "\u60a8\u597d\uff01\u5f88\u9ad8\u5174\u4e3a\u60a8\u89e3\u7b54\u5173\u4e8e\u53d1\u7968\u7684\u95ee\u9898\u54e6\u3002\n\n"
            "\u5173\u4e8e\u53d1\u7968\u670d\u52a1\uff0c\u4ee5\u4e0b\u4fe1\u606f\u4f9b\u60a8\u53c2\u8003\uff1a\n"
            "1. \u6211\u4eec\u7684\u6240\u6709\u5546\u54c1\u5747\u652f\u6301\u5f00\u5177\u6b63\u89c4\u53d1\u7968\uff0c\u53d1\u7968\u7c7b\u578b\u5305\u62ec\u589e\u503c\u7a0e\u666e\u901a\u53d1\u7968\uff08\u7535\u5b50\u7248\uff09\u548c\u589e\u503c\u7a0e\u4e13\u7528\u53d1\u7968\u3002\n"
            "2. \u7535\u5b50\u53d1\u7968\u4e00\u822c\u5728\u8ba2\u5355\u5b8c\u6210\u540e1-3\u4e2a\u5de5\u4f5c\u65e5\u5185\u5f00\u5177\uff0c\u60a8\u53ef\u5728\u8ba2\u5355\u8be6\u60c5\u4e2d\u67e5\u770b\u548c\u4e0b\u8f7d\u3002\n"
            "3. \u5982\u9700\u5f00\u5177\u589e\u503c\u7a0e\u4e13\u7528\u53d1\u7968\uff0c\u8bf7\u5728\u4e0b\u5355\u65f6\u586b\u5199\u6b63\u786e\u7684\u516c\u53f8\u540d\u79f0\u3001\u7eb3\u7a0e\u4eba\u8bc6\u522b\u53f7\u3001\u5730\u5740\u7535\u8bdd\u3001\u5f00\u6237\u884c\u53ca\u8d26\u53f7\u7b49\u4fe1\u606f\u3002\n"
            "4. \u53d1\u7968\u62ac\u5934\u5982\u586b\u5199\u9519\u8bef\uff0c\u53ef\u5728\u53d1\u7968\u5f00\u5177\u524d\u8054\u7cfb\u5ba2\u670d\u4fee\u6539\uff1b\u5982\u5df2\u5f00\u5177\uff0c\u9700\u7533\u8bf7\u4f5c\u5e9f\u91cd\u5f00\u3002\n\n"
            "\u5982\u9700\u7533\u8bf7\u53d1\u7968\u6216\u4fee\u6539\u53d1\u7968\u4fe1\u606f\uff0c\u5efa\u8bae\u60a8\u8054\u7cfb\u4eba\u5de5\u5ba2\u670d\u4e3a\u60a8\u529e\u7406\u54e6\uff01\n\n"
            "\u5e0c\u671b\u4ee5\u4e0a\u89e3\u7b54\u80fd\u591f\u5e2e\u5230\u60a8\uff0c\u82e5\u6709\u4efb\u4f55\u4e0d\u660e\u767d\u7684\u5730\u65b9\uff0c\u968f\u65f6\u6b22\u8fce\u60a8\u518d\u6765\u54a8\u8be2\u54e6\uff01\u795d\u60a8\u751f\u6d3b\u6109\u5feb\uff01"
        )
        return resp

    # \u6295\u8bc9\u76f8\u5173
    if any(kw in q for kw in ['\u6295\u8bc9', '\u5047\u8d27', '\u4e8c\u624b', '\u7ffb\u65b0', '\u865a\u5047\u5ba3\u4f20', '\u8fb1\u9a82', '\u6001\u5ea6\u5dee']):
        resp = (
            "\u60a8\u597d\uff01\u975e\u5e38\u62b1\u6b49\u7ed9\u60a8\u5e26\u6765\u4e86\u4e0d\u6109\u5feb\u7684\u4f53\u9a8c\uff0c\u60a8\u7684\u53cd\u9988\u6211\u4eec\u975e\u5e38\u91cd\u89c6\uff01\n\n"
            "\u5173\u4e8e\u60a8\u53cd\u6620\u7684\u95ee\u9898\uff0c\u6211\u4eec\u5efa\u8bae\u60a8\u91c7\u53d6\u4ee5\u4e0b\u6b65\u9aa4\uff1a\n"
            "1. \u8bf7\u4fdd\u7559\u597d\u76f8\u5173\u8bc1\u636e\uff08\u5982\u5546\u54c1\u7167\u7247\u3001\u804a\u5929\u8bb0\u5f55\u622a\u56fe\u3001\u5feb\u9012\u5355\u53f7\u3001\u5f00\u7bb1\u89c6\u9891\u7b49\uff09\uff0c\u8fd9\u4e9b\u5c06\u6709\u52a9\u4e8e\u6211\u4eec\u5c3d\u5feb\u6838\u5b9e\u5904\u7406\u3002\n"
            "2. \u8bf7\u60a8\u901a\u8fc7\u6211\u4eec\u7684\u5b98\u65b9\u552e\u540e\u670d\u52a1\u6e20\u9053\u63d0\u4ea4\u6b63\u5f0f\u6295\u8bc9\uff0c\u8be6\u7ec6\u63cf\u8ff0\u95ee\u9898\u5e76\u9644\u4e0a\u76f8\u5173\u51ed\u8bc1\u3002\n"
            "3. \u6211\u4eec\u4f1a\u5728\u6536\u5230\u6295\u8bc9\u540e\u76841-3\u4e2a\u5de5\u4f5c\u65e5\u5185\u5b89\u6392\u4e13\u4eba\u8ddf\u8fdb\u5904\u7406\uff0c\u5e76\u53ca\u65f6\u53cd\u9988\u5904\u7406\u8fdb\u5c55\u3002\n"
            "4. \u5982\u7ecf\u6838\u5b9e\u786e\u5c5e\u6211\u65b9\u8d23\u4efb\uff0c\u6211\u4eec\u5c06\u4e25\u683c\u6309\u7167\u76f8\u5173\u653f\u7b56\u4e3a\u60a8\u529e\u7406\u9000\u6362\u8d27\u3001\u9000\u6b3e\u6216\u8d54\u507f\u3002\n\n"
            "\u6211\u4eec\u5bf9\u7ed9\u60a8\u9020\u6210\u7684\u56f0\u6270\u6df1\u8868\u6b49\u610f\uff0c\u4e00\u5b9a\u4f1a\u8ba4\u771f\u5bf9\u5f85\u60a8\u7684\u6bcf\u4e00\u6761\u53cd\u9988\uff01\u5efa\u8bae\u60a8\u5c3d\u5feb\u8054\u7cfb\u4eba\u5de5\u5ba2\u670d\u63d0\u4ea4\u6295\u8bc9\u54e6\u3002\n\n"
            "\u5e0c\u671b\u4ee5\u4e0a\u89e3\u7b54\u80fd\u591f\u5e2e\u5230\u60a8\uff0c\u82e5\u6709\u4efb\u4f55\u4e0d\u660e\u767d\u7684\u5730\u65b9\uff0c\u968f\u65f6\u6b22\u8fce\u60a8\u518d\u6765\u54a8\u8be2\u54e6\uff01\u795d\u60a8\u751f\u6d3b\u6109\u5feb\uff01"
        )
        return resp

    # \u7269\u6d41/\u5feb\u9012\u76f8\u5173
    if any(kw in q for kw in ['\u5feb\u9012', '\u8fd0\u8d39', '\u7269\u6d41', '\u8865\u53d1', '\u7f3a\u4ef6', '\u5c11\u53d1', '\u5bc4\u5230\u56fd\u5916']):
        resp = (
            "\u60a8\u597d\uff01\u5f88\u9ad8\u5174\u4e3a\u60a8\u89e3\u7b54\u5173\u4e8e\u7269\u6d41\u914d\u9001\u7684\u95ee\u9898\u54e6\u3002\n\n"
            "\u5173\u4e8e\u60a8\u54a8\u8be2\u7684\u7269\u6d41\u95ee\u9898\uff0c\u4ee5\u4e0b\u4fe1\u606f\u4f9b\u60a8\u53c2\u8003\uff1a\n"
            "1. \u56fd\u5185\u6807\u51c6\u914d\u9001\u4e00\u822c\u4e3a3-5\u4e2a\u5de5\u4f5c\u65e5\uff0c\u504f\u8fdc\u5730\u533a\u53ef\u80fd\u9700\u8981\u989d\u59161-2\u5929\u3002\n"
            "2. \u5982\u6536\u5230\u5546\u54c1\u53d1\u73b0\u5c11\u4ef6\u6216\u7f3a\u4ef6\uff0c\u8bf7\u60a8\u5728\u7b7e\u6536\u540e48\u5c0f\u65f6\u5185\u8054\u7cfb\u5ba2\u670d\uff0c\u6211\u4eec\u4f1a\u5c3d\u5feb\u4e3a\u60a8\u5b89\u6392\u8865\u5bc4\u3002\n"
            "3. \u5982\u5feb\u9012\u5728\u8fd0\u8f93\u8fc7\u7a0b\u4e2d\u4e22\u5931\uff0c\u8bf7\u60a8\u53ca\u65f6\u8054\u7cfb\u5ba2\u670d\u5e76\u63d0\u4f9b\u8ba2\u5355\u53f7\uff0c\u6211\u4eec\u4f1a\u534f\u52a9\u60a8\u4e0e\u5feb\u9012\u516c\u53f8\u6838\u5b9e\u5e76\u5904\u7406\u3002\n"
            "4. \u6d77\u5916\u914d\u9001\u670d\u52a1\u8bf7\u54a8\u8be2\u4eba\u5de5\u5ba2\u670d\u4e86\u89e3\u5177\u4f53\u5730\u533a\u548c\u8fd0\u8d39\u8be6\u60c5\u3002\n\n"
            "\u5efa\u8bae\u60a8\u63d0\u4f9b\u8ba2\u5355\u53f7\uff0c\u8054\u7cfb\u4eba\u5de5\u5ba2\u670d\u4e3a\u60a8\u67e5\u8be2\u7269\u6d41\u72b6\u6001\u6216\u5904\u7406\u76f8\u5173\u95ee\u9898\u54e6\uff01\n\n"
            "\u5e0c\u671b\u4ee5\u4e0a\u89e3\u7b54\u80fd\u591f\u5e2e\u5230\u60a8\uff0c\u82e5\u6709\u4efb\u4f55\u4e0d\u660e\u767d\u7684\u5730\u65b9\uff0c\u968f\u65f6\u6b22\u8fce\u60a8\u518d\u6765\u54a8\u8be2\u54e6\uff01\u795d\u60a8\u751f\u6d3b\u6109\u5feb\uff01"
        )
        return resp

    # \u552e\u540e\u7ef4\u4fee\u76f8\u5173
    if any(kw in q for kw in ['\u7ef4\u4fee', '\u552e\u540e', '\u4e0a\u95e8\u5b89\u88c5', '\u5b89\u88c5\u670d\u52a1', '\u7ec8\u8eab\u7ef4\u4fee', '\u4fdd\u4fee']):
        resp = (
            "\u60a8\u597d\uff01\u5f88\u9ad8\u5174\u4e3a\u60a8\u89e3\u7b54\u5173\u4e8e\u552e\u540e\u670d\u52a1\u7684\u95ee\u9898\u54e6\u3002\n\n"
            "\u5173\u4e8e\u552e\u540e\u670d\u52a1\uff0c\u4ee5\u4e0b\u4fe1\u606f\u4f9b\u60a8\u53c2\u8003\uff1a\n"
            "1. \u6211\u4eec\u7684\u5546\u54c1\u5747\u4eab\u6709\u56fd\u5bb6\u89c4\u5b9a\u7684\u4e09\u5305\u670d\u52a1\uff08\u4fee\u7406\u3001\u66f4\u6362\u3001\u9000\u8d27\uff09\uff0c\u5177\u4f53\u4fdd\u4fee\u671f\u9650\u8bf7\u53c2\u8003\u5546\u54c1\u8be6\u60c5\u9875\u6216\u4ea7\u54c1\u4fdd\u4fee\u5361\u3002\n"
            "2. \u4fdd\u4fee\u8303\u56f4\u5185\u7684\u7ef4\u4fee\u670d\u52a1\u514d\u8d39\u63d0\u4f9b\uff0c\u4eba\u4e3a\u635f\u574f\u4e0d\u5728\u4fdd\u4fee\u8303\u56f4\u5185\uff0c\u7ef4\u4fee\u8d39\u7528\u6839\u636e\u5b9e\u9645\u635f\u574f\u60c5\u51b5\u8bc4\u4f30\u3002\n"
            "3. \u5982\u9700\u7533\u8bf7\u552e\u540e\u7ef4\u4fee\uff0c\u8bf7\u901a\u8fc7\u6211\u4eec\u7684\u552e\u540e\u670d\u52a1\u6e20\u9053\u63d0\u4ea4\u7533\u8bf7\uff0c\u5e76\u9644\u4e0a\u8d2d\u4e70\u51ed\u8bc1\uff08\u8ba2\u5355\u622a\u56fe\u6216\u53d1\u7968\uff09\u3002\n"
            "4. \u90e8\u5206\u5927\u5bb6\u7535\u5546\u54c1\u63d0\u4f9b\u514d\u8d39\u4e0a\u95e8\u5b89\u88c5\u670d\u52a1\uff0c\u5177\u4f53\u4ee5\u5546\u54c1\u8be6\u60c5\u9875\u8bf4\u660e\u4e3a\u51c6\u3002\n\n"
            "\u5efa\u8bae\u60a8\u8054\u7cfb\u4eba\u5de5\u5ba2\u670d\uff0c\u63d0\u4f9b\u5546\u54c1\u578b\u53f7\u548c\u95ee\u9898\u63cf\u8ff0\uff0c\u6211\u4eec\u4f1a\u4e3a\u60a8\u5b89\u6392\u4e13\u4e1a\u7684\u552e\u540e\u670d\u52a1\u54e6\uff01\n\n"
            "\u5e0c\u671b\u4ee5\u4e0a\u89e3\u7b54\u80fd\u591f\u5e2e\u5230\u60a8\uff0c\u82e5\u6709\u4efb\u4f55\u4e0d\u660e\u767d\u7684\u5730\u65b9\uff0c\u968f\u65f6\u6b22\u8fce\u60a8\u518d\u6765\u54a8\u8be2\u54e6\uff01\u795d\u60a8\u751f\u6d3b\u6109\u5feb\uff01"
        )
        return resp

    # \u5546\u54c1\u8d28\u91cf/\u4f7f\u7528\u95ee\u9898
    if any(kw in q for kw in ['\u8d28\u91cf', '\u4f7f\u7528\u4e00\u6b21', '\u574f\u4e86', '\u7455\u75b5', '\u5212\u75d5', '\u989c\u8272\u504f\u5dee', '\u548c\u56fe\u7247\u4e0d\u4e00\u6837', '\u7834\u635f', '\u5305\u88c5\u7834\u635f', '\u8fc7\u671f', '\u4fdd\u8d28\u671f']):
        resp = (
            "\u60a8\u597d\uff01\u975e\u5e38\u62b1\u6b49\u7ed9\u60a8\u5e26\u6765\u4e86\u4e0d\u597d\u7684\u4f7f\u7528\u4f53\u9a8c\uff0c\u6211\u4eec\u5bf9\u6b64\u6df1\u611f\u6b49\u610f\uff01\n\n"
            "\u5173\u4e8e\u60a8\u53cd\u6620\u7684\u5546\u54c1\u95ee\u9898\uff0c\u6211\u4eec\u5efa\u8bae\u60a8\uff1a\n"
            "1. \u8bf7\u4fdd\u7559\u597d\u5546\u54c1\u53ca\u5305\u88c5\uff0c\u5e76\u62cd\u6444\u6e05\u6670\u7684\u7167\u7247\u6216\u89c6\u9891\u4f5c\u4e3a\u51ed\u8bc1\u3002\n"
            "2. \u5982\u5546\u54c1\u57287\u5929\u5185\u51fa\u73b0\u8d28\u91cf\u95ee\u9898\uff0c\u652f\u6301\u9000\u6362\u8d27\u670d\u52a1\uff0c\u8fd0\u8d39\u7531\u6211\u4eec\u627f\u62c5\u3002\n"
            "3. \u5982\u8d85\u8fc77\u5929\u4f46\u5728\u4fdd\u4fee\u671f\u5185\uff0c\u53ef\u901a\u8fc7\u552e\u540e\u7ef4\u4fee\u6e20\u9053\u5904\u7406\u3002\n"
            "4. \u8f7b\u5fae\u5212\u75d5\u6216\u989c\u8272\u504f\u5dee\u5982\u4e0d\u5f71\u54cd\u6b63\u5e38\u4f7f\u7528\uff0c\u5efa\u8bae\u60a8\u8054\u7cfb\u5ba2\u670d\u534f\u5546\u89e3\u51b3\u65b9\u6848\u3002\n\n"
            "\u8bf7\u60a8\u901a\u8fc7\u552e\u540e\u670d\u52a1\u6e20\u9053\u63d0\u4ea4\u7533\u8bf7\u5e76\u9644\u4e0a\u76f8\u5173\u7167\u7247\uff0c\u5ba2\u670d\u4f1a\u5c3d\u5feb\u4e3a\u60a8\u5904\u7406\u54e6\uff01\n\n"
            "\u5e0c\u671b\u4ee5\u4e0a\u89e3\u7b54\u80fd\u591f\u5e2e\u5230\u60a8\uff0c\u82e5\u6709\u4efb\u4f55\u4e0d\u660e\u767d\u7684\u5730\u65b9\uff0c\u968f\u65f6\u6b22\u8fce\u60a8\u518d\u6765\u54a8\u8be2\u54e6\uff01\u795d\u60a8\u751f\u6d3b\u6109\u5feb\uff01"
        )
        return resp

    # \u4f18\u60e0\u5238/\u4ee5\u65e7\u6362\u65b0
    if any(kw in q for kw in ['\u4f18\u60e0\u5238', '\u4ee5\u65e7\u6362\u65b0']):
        resp = (
            "\u60a8\u597d\uff01\u5f88\u9ad8\u5174\u4e3a\u60a8\u89e3\u7b54\u5173\u4e8e\u4f18\u60e0\u6d3b\u52a8\u7684\u95ee\u9898\u54e6\u3002\n\n"
            "\u5173\u4e8e\u4f18\u60e0\u5238\u548c\u4ee5\u65e7\u6362\u65b0\u670d\u52a1\uff1a\n"
            "1. \u4f18\u60e0\u5238\u7684\u4f7f\u7528\u8303\u56f4\u548c\u6761\u4ef6\u8bf7\u4ee5\u9886\u53d6\u65f6\u7684\u8bf4\u660e\u4e3a\u51c6\uff0c\u90e8\u5206\u4f18\u60e0\u5238\u53ef\u80fd\u9650\u5b9a\u7279\u5b9a\u54c1\u7c7b\u6216\u91d1\u989d\u95e8\u69db\u3002\n"
            "2. \u4ee5\u65e7\u6362\u65b0\u670d\u52a1\u7684\u53ef\u7528\u8303\u56f4\u548c\u6298\u6263\u529b\u5ea6\uff0c\u8bf7\u54a8\u8be2\u4eba\u5de5\u5ba2\u670d\u4e86\u89e3\u6700\u65b0\u6d3b\u52a8\u8be6\u60c5\u3002\n"
            "3. \u4f18\u60e0\u5238\u4e00\u822c\u4e0d\u53ef\u53e0\u52a0\u4f7f\u7528\uff0c\u4e14\u4e0d\u4e0e\u5176\u4ed6\u4fc3\u9500\u6d3b\u52a8\u540c\u65f6\u4eab\u53d7\u3002\n\n"
            "\u5efa\u8bae\u60a8\u8054\u7cfb\u4eba\u5de5\u5ba2\u670d\u4e86\u89e3\u6700\u65b0\u7684\u4f18\u60e0\u6d3b\u52a8\u4fe1\u606f\u54e6\uff01\n\n"
            "\u5e0c\u671b\u4ee5\u4e0a\u89e3\u7b54\u80fd\u591f\u5e2e\u5230\u60a8\uff0c\u82e5\u6709\u4efb\u4f55\u4e0d\u660e\u767d\u7684\u5730\u65b9\uff0c\u968f\u65f6\u6b22\u8fce\u60a8\u518d\u6765\u54a8\u8be2\u54e6\uff01\u795d\u60a8\u751f\u6d3b\u6109\u5feb\uff01"
        )
        return resp

    # \u901a\u7528\u515c\u5e95
    resp = (
        "\u60a8\u597d\uff01\u611f\u8c22\u60a8\u7684\u54a8\u8be2\uff0c\u5f88\u9ad8\u5174\u4e3a\u60a8\u670d\u52a1\u54e6\u3002\n\n"
        "\u5173\u4e8e\u60a8\u54a8\u8be2\u7684\u95ee\u9898\uff0c\u5efa\u8bae\u60a8\u8054\u7cfb\u6211\u4eec\u7684\u5728\u7ebf\u4eba\u5de5\u5ba2\u670d\uff0c\u5ba2\u670d\u4f1a\u4e3a\u60a8\u63d0\u4f9b\u66f4\u8be6\u7ec6\u3001\u66f4\u51c6\u786e\u7684\u89e3\u7b54\u548c\u670d\u52a1\u3002\n"
        "\u60a8\u4e5f\u53ef\u4ee5\u62e8\u6253\u6211\u4eec\u7684\u5ba2\u670d\u70ed\u7ebf\u6216\u901a\u8fc7\u5b98\u65b9\u552e\u540e\u670d\u52a1\u6e20\u9053\u83b7\u53d6\u5e2e\u52a9\u3002\n\n"
        "\u518d\u6b21\u611f\u8c22\u60a8\u7684\u54a8\u8be2\uff0c\u795d\u60a8\u751f\u6d3b\u6109\u5feb\uff01"
    )
    return resp


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
    API_BASE_URL = "https://api.xiaomimimo.com/v1"
    CHAT_MODEL = "mimo-v2.5-pro"
    API_KEY = "sk-cvqrbdv498y5odkwgu88wq4rhgdv5z91bpukq5tu8mahw9yg"
    logger.info(f"RAG 运行模式：【小米 MiMo 官方 API】 | 当前模型：'{CHAT_MODEL}'")

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
        """使用小米高精度全模态大模型 mimo-v2-omni 将图像翻译成文字描述"""
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
        
        mimo_key = "sk-cvqrbdv498y5odkwgu88wq4rhgdv5z91bpukq5tu8mahw9yg"
        try:
            resp = requests.post(
                "https://api.xiaomimimo.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {mimo_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "mimo-v2-omni",
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
        """调用大模型，如果是纯文本模型，将通过多模态通道翻译为文字描述以实现穿透解析"""
        if images is None:
            images = []
            
        # 1. 针对纯文本大模型（如 deepseek / mimo-v2.5-pro），如果包含图片，则先通过多模态通道翻译成文本
        final_user_prompt = user_prompt
        is_pure_text = "deepseek" in CHAT_MODEL.lower() or "mimo-v2.5-pro" in CHAT_MODEL.lower()
        if images and is_pure_text:
            img_desc = RAGEngine.translate_image_to_text(images)
            final_user_prompt = f"{user_prompt}\n{img_desc}"
            # 翻译完毕后清空 images，避免向纯文本模型发送多模态结构引发 API 报错
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
        
            "空调手册": ["空调", "内机", "外机", "遥控器", "制冷", "制热", "除湿", "滤网", "百叶窗"],
            "冰箱手册": ["冰箱", "冷藏", "冻结", "温度", "变温"],
            "烤箱手册": ["烤箱", "烤箸", "烘焙", "预热", "温度"],
            "电钻手册": ["电钻", "充电", "电池", "钻头", "指示灯"],
            "健身追踪器手册": ["表带", "心率", "步数", "睡眠", "锁屏"],
            "发电机手册": ["发电机", "机油", "启动", "电压"],
            "洗碗机手册": ["洗碗机", "洗涤块", "亮碟剂", "餐具篮"],
            "空气净化器手册": ["净化器", "滤网", "灰尘传感器", "等离子"],
            "VR头显手册": ["VR", "头显", "显示屏", "遥控器"],
            "可编程温控器手册": ["温控器", "温度", "程序", "设定"],
            "人体工学椅手册": ["工学椅", "扶手", "气杆", "脚轮", "调节"],
            "蒸汽清洁机手册": ["蒸汽清洁", "喷淋臂", "清洁"],
            "健身单车手册": ["健身单车", "踢杆", "座垫", "脚踏"],
            "水泵手册": ["水泵", "流量", "扬程"],
            "摩托艇手册": ["摩托艇", "洛克机", "油门", "船尾", "活鱼舱"],
            "功能键盘手册": ["键盘", "按键", "功能键", "驱动程序"],
            "蓝牙激光鼠标手册": ["鼠标", "蓝牙", "配对", "电池"],
            "童电动摩托车手册": ["电动摩托车", "童车", "充电", "遥控器"],
            "维护器手册": ["维护器", "追踪"],
            "吹风机手册": ["吹风机", "风速", "u62a4罩"],
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
    is_eng = is_english_query(query)
            
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
        
    # 对含换行的复合问题，预先拆分为子问题再发送给 LLM
    pre_split_queries = []
    if chr(10) in query:
        parts = [p.strip().strip('"“”') for p in query.split(chr(10)) if p.strip()]
        if len(parts) >= 2:
            pre_split_queries = parts
        if len(parts) >= 2:
            pre_split_queries = parts
            logger.info(f"【Decompose】检测到复合问题，预拆分为 {len(parts)} 个子问题")
    
    effective_query = ' 。 '.join(pre_split_queries) if pre_split_queries else query
    user_prompt = f"当前问题: {effective_query}" if not is_eng else f"User query: {effective_query}"
    response = RAGEngine.call_multimodal_llm(system_prompt, user_prompt, [], history)
    
    try:
        cleaned = response.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"```json\s*|```\s*", "", cleaned)
        parsed = json.loads(cleaned.strip())
        
        # 扁平化并且清洗子查询，确保全部为纯文本字符串，防止 LLM 返回嵌套的 List 导致类型报错
        flat_queries = []
        def flatten_item(item):
            if isinstance(item, list):
                for sub_item in item:
                    flatten_item(sub_item)
            elif isinstance(item, str) and item.strip():
                flat_queries.append(item.strip())
                
        flatten_item(parsed)
        sub_queries = flat_queries
    except Exception:
        logger.warning(f"解析拆分问题失败，降级为原提问检索。")
        sub_queries = []
        
    # 将预拆分的子问题也加入 sub_queries
    for psq in pre_split_queries:
        if psq not in sub_queries:
            sub_queries.append(psq)
    
    if query not in sub_queries:
        sub_queries.insert(0, query)
        
    # 去重并保持顺序，限制最多5个子查询以减少噪声
    seen = set()
    sub_queries = [x for x in sub_queries if not (x in seen or seen.add(x))]
    if len(sub_queries) > 5:
        sub_queries = sub_queries[:5]
        
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
            stopwords = ["我想", "需要", "怎么", "如何", "哪里", "什么", "的", "了", "吗", "呢", "一个", "哪些", "这", "那", "有", "可选",
                  "请问", "您好", "哥", "姐", "小姐", "客服", "我们", "你们", "关于", "咨询", "了解", "查询", "详情", "具体"]
            clean_query = query
            for w in stopwords:
                clean_query = clean_query.replace(w, " ")
                
            raw_words = [w.strip() for w in re.split(r'[^\w\u4e00-\u9fa5]+', clean_query) if w.strip()]
            
            keywords = []
            for word in raw_words:
                if len(word) >= 3 and word not in sub_queries:
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
                
    # 使用 gte-rerank-v2 重排并筛选出 Top-20
    reranked_docs = RAGEngine.rerank_documents(query, merged_docs, top_n=15)
    
    logger.info("====== 【重排检索结果 Top-20 诊断日志】 ======")
    for idx, d in enumerate(reranked_docs):
        logger.info(f"Top-{idx+1} | 重排得分: {d['score']:.4f} | 标题: {d['section_title']} | 来自: 《{d['manual_name']}》 | 内容摘要: {d['content'][:80].replace(chr(10), ' ')}")
    logger.info("=========================================")
    
    return {"retrieved_docs": reranked_docs}

def generate_node(state: AgentState) -> Dict[str, Any]:
    query = state["query"]
    query_images = state["query_images"]
    history = state["history"]
    retrieved_docs = state["retrieved_docs"]
    
    is_eng = is_english_query(query)
    
    contexts_str = ""
    for i, doc in enumerate(retrieved_docs):
        img_tokens = [img["id"] for img in doc["images"]]
        if is_eng:
            contexts_str += (
                f"[Document-{i+1}] (Source: <{doc['manual_name']}>)\n"
                f"Content: {doc['content']}\n"
                f"Associated Image IDs: {img_tokens}\n"
                f"----------------------------------------\n"
            )
        else:
            contexts_str += (
                f"【文献-{i+1}】(出处: 《{doc['manual_name']}》)\n"
                f"内容: {doc['content']}\n"
                f"关联插图ID列表: {img_tokens}\n"
                f"----------------------------------------\n"
            )

    if is_eng:
        system_prompt = (
            "You are a professional product customer support assistant. Answer based on the [Retrieved Knowledge Base Content].\n"
            "\n"
            "[Rules]\n"
            "1. Quote verbatim: Copy the core content from retrieved docs exactly. Keep all line breaks and [IMAGE: xxx] tokens. Do not paraphrase specs or steps.\n"
            "2. Images inline: [IMAGE: xxx] must appear right after the relevant text, not at the end. Prefer docs that contain images.\n"
            "3. Model-specific: If user asks about a specific model (e.g., DCB107), only quote docs mentioning that model.\n"
            "4. Be concise: No lengthy greetings or closings. Product questions get direct answers. CS questions get warm, brief responses. No Markdown headers or numbered lists.\n"
            "5. Language: Entire response must be in English. Translate Chinese doc content to English. No Chinese characters.\n"
            "6. Unrelated content: Only say 'The retrieved knowledge base does not mention this content.' if nothing relevant found.\n"
            "7. Plain text only, no JSON or other formats.\n"
            "8. If you quote product text but relevant images are in other retrieved docs, append [IMAGE: image_id] at the end."
        )
        user_prompt = f"[Retrieved Knowledge Base Content]:\n{contexts_str}\n\nCurrent User Question: {query}"
    else:
        system_prompt = (
            "你是一个产品官方售后客服助手。请基于【检索知识库内容】回答用户提问。\n"
            "\n"
            "【核心规则】\n"
            "1. 直接引用原文：将检索到的核心内容原封不动抄录，保留所有换行符和 [IMAGE: xxx] 标记。不要自己改写参数和步骤。\n"
            "2. 图片紧跟内容：[IMAGE: xxx] 标记必须紧跟在相关文字后面，不要放到最后。多篇文献有的含图有的不含时，优先选含图的。\n"
            "3. 型号限定：用户提到具体型号（如DCB107），只能引用含该型号的文献。\n"
            "4. 简洁回答：不要加冗长问候和结语。产品问题直接给答案，客服问题温暖简洁回答。不要用Markdown标题和编号列表。\n"
            "5. 不相关内容：只有完全检索不到时才回答'知识库中未提及该内容。'\n"
            "6. 只输出纯文本，不要JSON或其他格式。\n"
            "7. 如果引用了产品描述但相关图片在其他文献中，请在答案末尾补充 [IMAGE: 图片ID]。"
        )
        user_prompt = f"【检索知识库内容】:\n{contexts_str}\n\n当前用户问题: {query}"
        
    answer = RAGEngine.call_multimodal_llm(system_prompt, user_prompt, query_images, history)
    return {"generated_answer": answer}

def verify_node(state: AgentState) -> Dict[str, Any]:
    query = state["query"]
    generated_answer = state["generated_answer"]
    retrieved_docs = state["retrieved_docs"]
    
    is_eng = is_english_query(query)
    
    contexts_str = ""
    for i, doc in enumerate(retrieved_docs):
        img_tokens = [img["id"] for img in doc["images"]]
        if is_eng:
            contexts_str += (
                f"[Document-{i+1}]\n"
                f"Content: {doc['content']}\n"
                f"Associated Image IDs: {img_tokens}\n"
                f"----------------------------------------\n"
            )
        else:
            contexts_str += (
                f"【文献-{i+1}】\n"
                f"内容: {doc['content']}\n"
                f"关联插图ID列表: {img_tokens}\n"
                f"----------------------------------------\n"
            )
        
    if is_eng:
        system_prompt = (
            "You are a professional text proofreading expert. Please verify if the [Preliminary Answer] is faithful to the [Reference Material].\n"
            "0. **Language Restriction**: The user's question and the preliminary answer are in English, but some [Reference Material] might be in Chinese. The preliminary answer's English translation of the Chinese reference is faithful and correct. Do NOT change it back to Chinese, and do NOT flag it as a conflict. The entire verified output MUST be in English. Absolutely do NOT output any Chinese characters in your response!\n"
            "1. Only correct factual errors that explicitly conflict with the Reference. If the preliminary answer has no factual conflicts, you MUST keep the preliminary answer 100% exactly as it is, word for word!\n"
            "2. [STRICTLY FORBID ADDING INFORMATION]: You must NOT add or append any new sections, sentences, or extra information that was not in the [Preliminary Answer]! Even if it exists in the Reference, do not add it!\n"
            "3. Keep all '[IMAGE: xxx]' tokens and original line breaks exactly as they are in the answer.\n"
            "4. Output ONLY the final verified answer text. Do not add any proofreading explanations or extra words."
        )
        user_prompt = f"[Reference Material]:\n{contexts_str}\n\n[User Question]: {query}\n\n[Preliminary Answer]: {generated_answer}"
    else:
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
    
    if is_eng:
        denial_keywords = ["not mention", "not found", "no information", "cannot find", "unable to answer", "unrelated"]
    else:
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

        is_eng = is_english_query(query)

        # ====== 中文通用客服问题直接走模板，跳过 RAG ======
        if not is_eng and classify_zh_generic(query):
            logger.info(f"【分类器】检测到中文通用客服问题，使用模板响应。问题: '{query[:50]}'")
            return get_zh_template_response(query)

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
            ans = result.get("final_answer", "")
            if not ans:
                if is_eng:
                    ans = "The retrieved knowledge base does not mention this content."
                else:
                    ans = "知识库中未提及该内容。"
            
            if is_eng:
                denial_keywords = ["not mention", "not found", "no information", "cannot find", "unable to answer", "unrelated"]
            else:
                denial_keywords = ["未提及该内容", "未提及", "没有提及", "无法回答", "知识库中未找到"]
                
            if any(k in ans.lower() for k in denial_keywords) or len(ans.strip()) < 15:
                if is_eng:
                    core = query.strip("? ").replace("please tell me", "").replace("how to", "").replace("how do I", "").replace("What is", "").replace("What are", "")
                    if len(core) > 35:
                        core = core[:35] + "..."
                    import random
                    templates = [
                        f"Hello! Regarding your inquiry about '{core}', I'm very sorry, but I couldn't find any information or mention of this in our official product manuals or knowledge base. I recommend reaching out directly to our live customer support team for further assistance. Thank you so much for your understanding!",
                        f"Hello! I'd be happy to help you. I sincerely apologize for any inconvenience, but we couldn't find any information regarding '{core}' in our product documentation. Please contact our live customer service team, and we will do our best to assist you!",
                        f"Hello! Concerning your question about '{core}', I have carefully checked our official manuals, but there doesn't seem to be any detail about this in our knowledge base. Please feel free to connect with our online customer support team. We are always here to help!"
                    ]
                    ans = random.choice(templates)
                else:
                    # 剔除提问中的语气词与通用句式以抓取核心意图
                    core = query.strip("？?吗呢哈").replace("请问", "").replace("我想咨询", "").replace("我想了解", "").replace("你们家的", "").replace("你们的", "")
                    if len(core) > 25:
                        core = core[:25] + "..."
                    import random
                    templates = [
                        f"您好！关于您咨询的“{core}”问题，非常抱歉，在我们的官方产品使用手册和知识库中暂时没有找到相关的记录或提及哦。建议您直接联系我们官方在线售后客服为您进行进一步核实与处理，非常感谢您的理解哈！",
                        f"您好！很高兴为您服务哈。非常抱歉给您带来困扰，关于您咨询的“{core}”，在我们的产品手册知识库中确实未提及这部分内容。建议您点击联系人工客服，我们会第一时间为您核实解答的哦！",
                        f"您好！关于您提到的“{core}”事宜，我仔细为您查找了官方的产品手册，里面暂时没有提到这方面的详细规定或信息呢。建议您随时联系官方在线售后团队，我们定会竭诚为您服务的哈！"
                    ]
                    ans = random.choice(templates)
            
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
                                    
                if is_eng:
                    multimodal_keywords = ["image", "picture", "illustration", "figure", "light", "flash", "led", "strap", "size", "install", "position", "structure", "front", "back", "interface", "caster", "gas lift", "armrest", "console", "display", "dial"]
                    denial_check = "not mention"
                    pic_msg = " Please refer to the following related illustrations: <PIC>"
                else:
                    multimodal_keywords = ["图", "闪", "指示灯", "灯", "表带", "尺寸", "安装", "位置", "结构", "正面", "背面", "接口", "脚轮", "气杆", "扶手", "控制台", "显示屏", "表盘"]
                    denial_check = "未提及该内容"
                    pic_msg = " 请参考以下相关插图：<PIC>"
                    
                if fallback_images and any(k in query_lower or k in ans.lower() for k in multimodal_keywords):
                    if denial_check not in ans.lower():
                        if "<PIC>" not in ans:
                            ans += pic_msg
                        images_used = fallback_images[:3]  # 最多取前3张以防止多图异常
            
            # 再次确保回答文本中有且至少有一个 <PIC>
            if images_used:
                if "<PIC>" not in ans:
                    if is_eng:
                        ans += " Please refer to the following related illustrations: <PIC>"
                    else:
                        ans += " 请参考以下相关插图：<PIC>"
                ans += f', {json.dumps(images_used)}'
                
            return ans
        except Exception as e:
            logger.error(f"LangGraph 运行异常: {str(e)}")
            if is_eng:
                return "The service has encountered an error. Please try again later."
            else:
                return "服务出现故障，请稍后重试。"

if __name__ == "__main__":
    agent = ConversationalAgent()
    ans = agent.ask("测试一下大模型连通性")
    print(ans)
