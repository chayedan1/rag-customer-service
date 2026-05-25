# -*- coding: utf-8 -*-
import os
import re
import csv
import pandas as pd

def is_english_query(query: str) -> bool:
    if not isinstance(query, str):
        return False
    zh_chars = len(re.findall(r'[\u4e00-\u9fff]', query))
    en_words = len(re.findall(r'[a-zA-Z]{2,}', query))
    if zh_chars == 0 and en_words > 0:
        return True
    return en_words > zh_chars

def remove_emojis_and_non_bmp(text):
    if not isinstance(text, str):
        return ""
    # Filter out characters outside the Basic Multilingual Plane (BMP)
    return "".join(c for c in text if ord(c) <= 0xffff)

def clean_and_format_submission():
    input_file = "submission.csv"
    questions_file = "question_public.csv"
    output_utf8 = "submission.csv"
    output_bom = "submission_utf8_sig.csv"
    output_gbk = "submission_gbk.csv"
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found!")
        return
        
    if not os.path.exists(questions_file):
        print(f"Error: {questions_file} not found!")
        return
        
    print(f"Reading generated submission: {input_file}")
    df_s = pd.read_csv(input_file)
    print(f"Reading questions: {questions_file}")
    df_q = pd.read_csv(questions_file, encoding='utf-8')
    
    # Create question map
    q_map = dict(zip(df_q['id'], df_q['question']))
    
    print(f"Original shape: {df_s.shape}")
    
    # 1. Deduplicate by 'id' and sort by 'id' ascending
    df_s = df_s.drop_duplicates(subset=["id"]).sort_values(by="id")
    print(f"Deduplicated and sorted shape: {df_s.shape}")
    
    # 2. Assert we have exactly the right amount of rows (e.g. 400 questions)
    if len(df_s) != 400:
        print(f"Warning: Expected 400 rows, but got {len(df_s)} rows.")
        
    cleaned_answers = []
    for idx, row in df_s.iterrows():
        qid = int(row['id'])
        ans = str(row['ret'])
        
        # Get original question
        q_text = q_map.get(qid, "")
        is_eng = is_english_query(q_text)
        
        # 精准匹配行尾可能附带的图片 JSON 数组, 如: `, ["Manual04_12", "Blower_01"]`
        pattern = r'(,\s*\[\s*"[^"]+"\s*(?:,\s*"[^"]+"\s*)*\])\s*$'
        match = re.search(pattern, ans)
        if match:
            img_part = match.group(1).strip()
            main_text = ans[:match.start()]
        else:
            img_part = ""
            main_text = ans
            
        # A. 移除非 BMP 字符和表情符号（只针对主文本）
        main_text = remove_emojis_and_non_bmp(main_text)
        
        # B. 处理换行符，全部转义为字面量 \\n
        main_text = main_text.replace('\r\n', '\\n').replace('\n', '\\n').replace('\r', '\\n')
        
        # C. 清洗主文本中的一切英文单/双引号以及反引号，保证 CSV 极其安全
        main_text = main_text.replace('"', '').replace("'", "").replace("`", "")
        
        if is_eng:
            # D. 处理英文问题：翻译和剥离中文字符
            main_text = main_text.replace("汇总英文手册", "English Manual").replace("汇总中文手册", "Chinese Manual")
            main_text = main_text.replace("文献", "Document")
            # 剥离任何可能残存的中文
            main_text = re.sub(r'[\u4e00-\u9fff]', '', main_text)
            # 将主文本里的所有半角英文逗号替换为分号，防止格式串扰
            main_text = main_text.replace(",", ";")
        else:
            # E. 处理中文问题：将主文本里的所有半角英文逗号替换为全角逗号，保证主文本中绝无英文逗号
            main_text = main_text.replace(",", "，")
            
        # F. 将清洗后的正文与原样保留的图片 JSON 数组重新拼合
        ans_cleaned = main_text.strip() + img_part
        cleaned_answers.append(ans_cleaned)
        
    df_s['ret'] = cleaned_answers
    
    # 备份原始文件
    backup_file = "submission_backup_raw_multiline.csv"
    if not os.path.exists(backup_file):
        import shutil
        shutil.copyfile(input_file, backup_file)
        print(f"Backed up raw multiline submission to: {backup_file}")
        
    # 写入 CSV 文件，采用默认引用机制（不使用 QUOTE_NONE，由 pandas 自动智能加双引号保护含逗号/引号的图片列）
    df_s.to_csv(output_utf8, index=False, encoding='utf-8')
    print(f"Successfully saved clean submission to (UTF-8): {output_utf8}")
    
    # 写入带 BOM 的 UTF-8（解决 Windows Excel 乱码）
    df_s.to_csv(output_bom, index=False, encoding='utf-8-sig')
    print(f"Successfully saved clean submission to (UTF-8 with BOM): {output_bom}")
    
    # 写入 GBK（备用）
    try:
        df_s.to_csv(output_gbk, index=False, encoding='gbk', errors='replace')
        print(f"Successfully saved clean submission to (GBK): {output_gbk}")
    except Exception as e:
        print(f"Could not save in GBK format: {e}")
        
    # 强力校验物理行数与多模态格式
    for filename in [output_utf8, output_bom, output_gbk]:
        if os.path.exists(filename):
            try:
                with open(filename, 'r', encoding='utf-8-sig' if 'sig' in filename else ('gbk' if 'gbk' in filename else 'utf-8'), errors='ignore') as f:
                    lines = f.readlines()
                print(f"Verification: {filename} has exactly {len(lines)} raw lines.")
            except Exception as e:
                print(f"Could not verify {filename}: {e}")

if __name__ == "__main__":
    clean_and_format_submission()
