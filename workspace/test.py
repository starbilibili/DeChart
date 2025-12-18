import torch
import os
import sys
import json
import time
import csv
import pdb
import shutil
import argparse
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm
from copy import deepcopy
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration,Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from evaluation import evaluate_internvl_in_chartqa

os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# 对结果的必要后处理
def response_postprocessing(response, separator):
    if not response:
        return []
    if len(response.split(separator)) == 1:
        return response
    response = response.strip().split(separator)
    if 'START' in response[0]:
        response = response[1:]
    if 'END' in response[-1]:
        response = response[:-1]
    response = [i.strip() for i in response if len(i) > 0]
    return response
def qwen_postprocess(text):
    # 提取表格数据
    lines = text.strip().split('\n')  # 按行分割
    table_data = [line.strip().split('|')[1:-1] for line in lines if '|' in line]  # 提取表格内容
    table_data = [[item.strip() for item in row] for row in table_data]  # 去除多余空格

    return table_data


def is_badcase(response):
    if len(response) <= 1 or len(response) >= 15:
        return True
    for r in response:
        if len(r.split(',')) > 10:
            return True
    return False
    
def get_messages_single_chat(image_path, prompt):
    messages_template = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "https://modelscope.oss-cn-beijing.aliyuncs.com/resource/qwen.png",
                },
                {"type": "text", "text": "Generate underlying data table of the figure"},
            ],
        },
    ]
    question_text = prompt['messages'][0]['content']
    messages_content = messages_template[1]["content"]
    messages_content[0]['image'] = image_path
    messages_content[1]['text'] = question_text
    messages_template[1]["content"] = messages_content
    return messages_template

def get_messages_multi_chat(image_path, prompt):
    messages1 = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "https://modelscope.oss-cn-beijing.aliyuncs.com/resource/qwen.png",
                },
                {"type": "text", "text": "Generate underlying data table of the figure"},
                {"type": "text", "text": "Generate underlying data table of the figure"},
            ],
        },
    ]
    messages2 = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Generate underlying data table of the figure"},
            ],
        },
    ]
    system = prompt['messages'][0]['content']
    question1 = prompt['messages'][1]['content']
    question2 = prompt['messages'][3]['content']
    messages1[0]['content'] = system
    messages1[1]['content'][0]['image'] = image_path
    messages1[1]['content'][1]['text'] = question1
    messages1[1]['content'][2]['text'] = question2
    return messages1

if __name__ == '__main__':
    argparse = argparse.ArgumentParser()
    argparse.add_argument('--model_path', type=str, help='Path to the model')
    argparse.add_argument('--saved_dir', type=str, help='Path to the output directory')
    args = argparse.parse_args()
    model_path = args.model_path
    # model_path = "/data5/home/xiechenyu2023/project/ChartQA/Qwen2/pretrained_model/Qwen2-VL-2B-Instruct"
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_path)

    test_dir = "/data5/home/xiechenyu2023/project/ChartQA/mydatasets/PlotQA_test_with_CoT.jsonl"
    saved_dir = args.saved_dir
    # 如果文件夹不存在，则创建
    if not os.path.exists(f"{saved_dir}"):
        os.makedirs(f"{saved_dir}")
    # 如果文件夹存在，则删除其中的所有文件，文件夹本身不删除
    else:
        shutil.rmtree(f"{saved_dir}")
        os.makedirs(f"{saved_dir}")

    with open(test_dir, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        lines = [json.loads(line) for line in lines]

    for line in tqdm(lines[::], file=sys.stdout):
        image_path = line['images'][0]
        image_name = image_path.split('images/')[-1]
        file_name = image_name.split('.')[0]
        # messages = get_messages_single_chat(image_path, line)
        messages = get_messages_multi_chat(image_path, line)
        # texts = [
        #     processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        #     for msg in messages
        # ]
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        generated_ids = model.generate(**inputs, max_new_tokens=1024)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        print(output_text)
        # response = qwen_postprocess(output_text[0])
        response = response_postprocessing(output_text[0], separator='<0x0A>')
        print(response)
        output_path = f"{saved_dir}/{file_name}.csv"
        # 创建output_path的文件夹
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))
        with open(output_path, 'w') as f:
            # writer = csv.writer(f)
            # writer.writerows(response)
            for i in response:
                f.write(i + '\n')
