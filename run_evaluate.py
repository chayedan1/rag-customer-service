# -*- coding: utf-8 -*-
"""运行批量评测的脚本（保守并发数，防止 API 限频）"""
import os
os.environ["PYTHONUTF8"] = "1"

from evaluate import BatchEvaluator

if __name__ == "__main__":
    evaluator = BatchEvaluator(max_workers=3)
    evaluator.run()
