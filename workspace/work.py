import json
import os
import pandas as pd
import csv

# 配置路径
source_path = "/data5/home/xiechenyu2023/project/ChartQA/mydatasets/ChartQA/test/test_human.json"
output_path = "/data5/home/xiechenyu2023/project/ChartQA/mydatasets/ChartQA/test/Qwen_QA_test_human_trainfree.json"
image_root = "/data5/home/xiechenyu2023/project/ChartQA/mydatasets/ChartQA/test/png"
table_root = "/data5/home/xiechenyu2023/project/ChartQA/mydatasets/ChartQA/test/tables"

# 读取原始数据
with open(source_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 转换格式
converted_data = []
failed_tables = []  # 记录哪些表格读取失败

def df_to_markdown(df):
    if df.empty:
        return ""
    # 表头
    header = "| " + " | ".join(df.columns) + " |"
    # 分隔行
    separator = "| " + " | ".join(["---"] * len(df.columns)) + " |"
    # 数据行
    rows = ["| " + " | ".join(str(cell) for cell in row) + " |" for row in df.values]
    return "\n".join([header, separator] + rows)

for item in data:
    imgname = item["imgname"]
    query = item["query"]
    label = item["label"]

    # 生成表格文件名：xxx.png -> xxx.csv
    table_filename = os.path.splitext(imgname)[0] + ".csv"
    table_path = os.path.join(table_root, table_filename)

    # 读取表格内容
    table_content = ""
    if os.path.exists(table_path):
        try:
            df = pd.read_csv(table_path)
            # 转为易读的字符串格式（保留表头和对齐）
            table_content = df_to_markdown(df)
        except Exception as e:
            print(f"⚠️ 读取表格失败 {table_path}: {e}")
            failed_tables.append(table_path)
            table_content = "[TABLE_LOAD_FAILED]"
    else:
        print(f"⚠️ 表格文件不存在: {table_path}")
        failed_tables.append(table_path)
        table_content = "[TABLE_NOT_FOUND]"

    # 构造用户提示（英文 + 填入表格内容）
    user_prompt = (
        f"<image>\n"
        f"{query}\n\n"
        f"\nPlease first reason step by step, then put your final answer within <answer> tags like this: <answer>your final answer</answer>.\n"
        f"Example format:\n"
        f"thinking process. <answer>Final Answer Here</answer>\n"
    )

    # 构造 messages
    messages = [
        {
            "role": "user",
            "content": user_prompt
        },
        {
            "role": "assistant",
            "content": label
        }
    ]

    # 图像路径
    image_path = os.path.join(image_root, imgname)

    # 组合成目标格式
    converted_item = {
        "messages": messages,
        "images": [image_path]
    }
    converted_data.append(converted_item)

# 保存转换后的数据
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(converted_data, f, indent=2, ensure_ascii=False)

print(f"✅ 转换完成，共处理 {len(converted_data)} 条数据，已保存至：{output_path}")
if failed_tables:
    print(f"⚠️ 共有 {len(failed_tables)} 个表格文件读取失败或不存在。")