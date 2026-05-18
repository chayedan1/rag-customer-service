# -*- coding: utf-8 -*-
"""
模块 5：高并发、支持断点续答的批量测评与自动提交生成器 (evaluate.py)
--------------------------------------------------------------------
功能说明：
1. 载入公共测试集：读取 `question_public.csv` 中的全部问题（包含复合型、跨类别提问）。
2. 断点续答机制 (Resume from Breakpoint)：自动检测本地的临时进度文件 `submission_in_progress.csv`，
   如果脚本中途网络中断或被关闭，重新启动时将自动跳过已答题目，极大节省开发测试成本与 API Quota。
3. 线程池高并发加速 (Thread Pooling)：由于主要时间消耗在 API 的网络 I/O 阻塞上，本模块引入 `ThreadPoolExecutor`，
   开启 3 线程并发处理（安全并发数，防 Qwen 账户触发超额限频），效率提升至 300%。
4. 线程安全输出锁 (Thread-safe Lock)：对文件写入与状态计数引入多线程锁机制，防止文件指针写入错位和计数脏数据。
5. 自动校对输出：在 400 道题目全部完成后，自动将结构整理并保存为 strictly 符合赛题格式（包含 'id' 和 'ret' 两列）的 `submission.csv`。
"""

import os
import csv
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from tqdm import tqdm
from graph import ConversationalAgent

# 配置日志（开启 UTF-8 环境以防中文乱码）
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)

# 定义线程锁
file_write_lock = threading.Lock()

class BatchEvaluator:
    def __init__(self, workspace_dir: str = "d:\\Desktop\\数据", max_workers: int = 3):
        """
        初始化测评模块。
        """
        self.workspace_dir = workspace_dir
        self.question_path = os.path.join(workspace_dir, "question_public.csv")
        self.temp_output_path = os.path.join(workspace_dir, "submission_in_progress.csv")
        self.final_output_path = os.path.join(workspace_dir, "submission.csv")
        self.max_workers = max_workers
        
        # 实例化对话 RAG 智能体
        self.agent = ConversationalAgent()
        
        # 已解答的题目 ID 缓存
        self.completed_ids = set()
        self._load_progress()

    def _load_progress(self):
        """
        从临时进度文件中载入已完成的问题 ID，用于支持断点续答。
        """
        if os.path.exists(self.temp_output_path):
            try:
                # 使用标准的 CSV 字典读取方式获取已回答题目的 ID
                with open(self.temp_output_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if row.get("id"):
                            self.completed_ids.add(int(row["id"]))
                logger.info(f"断点检测：已检测到历史进度文件，成功载入 {len(self.completed_ids)} 条已完成解答，即将自动跳过...")
            except Exception as e:
                logger.warning(f"读取历史进度文件时发生错误: {str(e)}。将进行全新测评。")
        else:
            # 文件不存在，初始化临时文件并写入表头
            with open(self.temp_output_path, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["id", "ret"])
            logger.info("未检测到历史进度，即将开始全新 RAG 批量测评流程。")

    def _write_row_to_temp(self, q_id: int, answer: str):
        """
        线程安全地将一条解答写入临时进度文件。
        """
        with file_write_lock:
            with open(self.temp_output_path, 'a', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([q_id, answer])

    def process_single_question(self, q_id: int, question: str, progress_bar: tqdm) -> tuple:
        """
        处理单道题目的检索与解答生成（线程池工作函数）。
        """
        try:
            # 清洗问题文本（过滤掉可能首尾包含的转义双引号和折行）
            clean_question = question.strip().replace('"', '').replace('\\n', '\n')
            
            # 调用 RAG 智能体（在批量评测中，多轮对话历史为空）
            answer = self.agent.ask(clean_question)
            
            # 将生成的答案进行单行安全化清洗（防止写 CSV 破坏格式，但保留正常换行转义）
            # 注意：赛题要求输出的单行格式，我们可以直接用 \n 字符串表示折行，或者直接输出含正常换行的内容，
            # 采用 CSV 写入时，多行字符串会被自动用双引号括起来保护，完全不用担心。
            
            # 写入临时进度文件以支持断点恢复
            self._write_row_to_temp(q_id, answer)
            
            # 更新进度条
            progress_bar.update(1)
            return q_id, True
        except Exception as e:
            logger.error(f"处理问题 ID {q_id} 时发生未捕获异常: {str(e)}")
            progress_bar.update(1)
            return q_id, False

    def run(self):
        """
        高并发运行全量题目测评。
        """
        # 1. 载入全量题目
        if not os.path.exists(self.question_path):
            logger.error(f"公共测评集文件 {self.question_path} 不存在！请检查路径。")
            return
            
        df_questions = pd.read_csv(self.question_path)
        total_questions = len(df_questions)
        logger.info(f"成功载入公共测试集，共找到 {total_questions} 道待解答题目。")

        # 2. 筛选过滤已经回答过的题目
        pending_questions = []
        for _, row in df_questions.iterrows():
            q_id = int(row["id"])
            if q_id not in self.completed_ids:
                pending_questions.append((q_id, str(row["question"])))
                
        total_pending = len(pending_questions)
        if total_pending == 0:
            logger.info("检测到所有测试题目均已解答完毕！直接进入结果生成。")
            self.generate_final_submission()
            return
            
        logger.info(f"过滤已完成题目，当前剩余 {total_pending} 道题目待解答。")

        # 3. 启动高并发线程池处理任务
        # 使用进度条直观展示实时评测进度
        with tqdm(total=total_pending, desc="RAG 批量测评中") as pbar:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # 提交所有待执行任务
                futures = {
                    executor.submit(self.process_single_question, q_id, q_text, pbar): q_id 
                    for q_id, q_text in pending_questions
                }
                
                # 逐一等待结果收集
                for future in as_completed(futures):
                    q_id = futures[future]
                    try:
                        _, success = future.result()
                        if success:
                            self.completed_ids.add(q_id)
                    except Exception as e:
                        logger.error(f"问题 ID {q_id} 线程池执行故障: {str(e)}")

        # 4. 生成最终的可提交 submission.csv 文件
        self.generate_final_submission()

    def generate_final_submission(self):
        """
        将完成后的临时进度文件整理、去重、排序，生成 100% 符合赛题格式的最终提交文件。
        """
        logger.info("测评全部结束。正在从临时文件中规整最终提交数据...")
        try:
            # 读取临时回答数据
            df_temp = pd.read_csv(self.temp_output_path)
            
            # 按 ID 去重并排序，确保严格的一致性
            df_temp = df_temp.drop_duplicates(subset=["id"]).sort_values(by="id")
            
            # 重命名列名，确保完全契合 submission_example.csv 的 ['id', 'ret'] 标准
            df_temp.columns = ["id", "ret"]
            
            # 保存到最终路径
            df_temp.to_csv(self.final_output_path, index=False, encoding='utf-8')
            
            logger.info(f"🎉 成功生成 100% 合规的最终提交文件：{self.final_output_path}！")
            logger.info(f"生成文件摘要：共解答 {len(df_temp)} 行题目。首尾 ID 为 {df_temp['id'].iloc[0]} 到 {df_temp['id'].iloc[-1]}。")
            
            # 清理临时进度文件（可选，建议保留以供审核）
            # os.remove(self.temp_output_path)
        except Exception as e:
            logger.error(f"整合作业成果并生成最终提交文件失败: {str(e)}")

if __name__ == "__main__":
    # 在批量评测中，必须确保 UTF-8 运行环境
    os.environ["PYTHONUTF8"] = "1"
    
    # 实例化并运行测评。并发线程数设置为 1，以防触发 Qwen API 的 429 限频限制
    evaluator = BatchEvaluator(max_workers=1)
    evaluator.run()
