import argparse
import json
import os
import shutil
import sys
from copy import deepcopy
from tqdm import tqdm
import torch
from transformers import AutoProcessor
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from functools import partial
# 添加缺失的导入
from transformers import Qwen2_5_VLForConditionalGeneration


def get_messages_multi_chat(image_path, prompt):
    system = prompt['messages'][0]['content']
    question1 = prompt['messages'][1]['content']
    question2 = prompt['messages'][3]['content']
    
    messages = [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": question1},
                {"type": "text", "text": question2},
            ],
        },
    ]
    return messages

# 自定义数据集类
class VisionDataset(Dataset):
    def __init__(self, data, processor):
        self.data = data
        self.processor = processor
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        line = self.data[idx]
        image_path = line['images'][0]
        file_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # 创建消息
        messages = get_messages_multi_chat(image_path, line)
        
        # 应用模板并获取文本
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        # 打开并处理图像
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error opening image {image_path}: {e}")
            # 返回空图像作为占位符
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        return {
            "text": text,
            "image": image,
            "image_path": image_path,
            "file_name": file_name
        }


def collate_fn(batch, processor):
    texts = [item["text"] for item in batch]
    images = [item["image"] for item in batch]
    image_paths = [item["image_path"] for item in batch]
    file_names = [item["file_name"] for item in batch]
    
    # 处理输入
    inputs = processor(
        text=texts,
        images=images,
        padding=True,
        return_tensors="pt",
    )
    
    return inputs, image_paths, file_names

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
    parser.add_argument('--saved_dir', type=str, required=True, help='Path to the output directory')
    parser.add_argument('--test_file', type=str, required=True, help='Path to test JSONL file')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for inference')
    args = parser.parse_args()
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # 创建输出目录
    os.makedirs(args.saved_dir, exist_ok=True)
    
    # 加载模型和处理器
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(args.model_path, use_fast=True)
    
    # 读取测试数据
    with open(args.test_file, 'r', encoding='utf-8') as f:
        lines = [json.loads(line) for line in f.readlines()]
    
    # 创建数据集和数据加载器
    dataset = VisionDataset(lines, processor)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=partial(collate_fn, processor=processor),
        num_workers=min(8, os.cpu_count()),  # 使用所有可用核心
        shuffle=False,
        pin_memory=True  # 加速数据传输到GPU
    )
    
    # 批量处理
    for batch in tqdm(dataloader, desc="Processing batches"):
        inputs, image_paths, file_names = batch
        
        # 将输入移到GPU
        inputs = inputs.to("cuda")
        
        # 生成输出
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=1024,
                pad_token_id=processor.tokenizer.pad_token_id
            )
        
        # 去除输入部分，只保留生成的文本
        input_lengths = inputs.input_ids.shape[1]
        generated_texts = processor.batch_decode(
            [ids[input_lengths:] for ids in generated_ids],
            skip_special_tokens=True
        )
        
        # 处理并保存每个样本的结果
        for text, file_name in zip(generated_texts, file_names):
            response = response_postprocessing(text, separator='<0x0A>')
            print(response, flush=True)
            with open(f"{args.saved_dir}/{file_name}.csv", 'w') as f:
                for line in response:
                    f.write(line + '\n')
    
    print(f"Processing completed. Results saved to {args.saved_dir}")