import os
import sys
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
# os.environ["WANDB_API_KEY"] = 'YOUR_API_KEY' 
os.environ["WANDB_DISABLED"] = "true"
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

# 调用训练命令
os.system("llamafactory-cli train examples/train_lora/qwen2vl_lora_sft.yaml")
