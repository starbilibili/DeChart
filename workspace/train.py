import os
import sys
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
# os.environ["WANDB_API_KEY"] = 'e3848d16484439694892e4f30d7e85fb1a67e482' 
os.environ["WANDB_DISABLED"] = "true"
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

# 2. 添加路径映射（根据实际路径修改）
transformers_path = "/data5/home/xiechenyu2023/project/ChartQA/Qwen2/transformers-4.49.0/src"
sys.path = [transformers_path] + sys.path
# 4. 确保transformers库被强制重新加载
def reload_transformers():
    """强制重新加载transformers模块"""
    import importlib
    modules_to_reload = [m for m in sys.modules if m.startswith("transformers")]
    for module in modules_to_reload:
        importlib.reload(sys.modules[module])
reload_transformers()
print(os.path.isdir(transformers_path))
# 调用训练命令
os.system("llamafactory-cli train examples/train_lora/qwen2vl_lora_sft.yaml")