# 设置 CUDA 设备
export CUDA_VISIBLE_DEVICES=0,3

# 定义变量
MODEL_ROOT="/lustre/home/xiechenyu2023/saved_model/qwen2.5"  # 模型根目录
OUTPUT_ROOT="/data5/home/xiechenyu2023/project/ChartQA/ChartDetect/a-workspace/output_dir/qwen_results"  # 输出结果根目录
TEST_FILE="/data5/home/xiechenyu2023/project/ChartQA/mydatasets/ChartQA_test_with_CoT.jsonl"
# 单模型测试
MODEL_NAME=qwen2.5_vl-7b_ChartQA_CoT_checkpoint-23000
# 构建完整的模型路径
MODEL_PATH="$MODEL_ROOT/$MODEL_NAME"
# 构建输出目录名称
SAVED_DIR="$OUTPUT_ROOT/$MODEL_NAME"

# 打印当前处理的模型信息
echo "Processing model: $MODEL_NAME"
echo "Model path: $MODEL_PATH"
echo "Output directory: $SAVED_DIR"

# 执行 test.py 脚本
python /data5/home/xiechenyu2023/project/ChartQA/Qwen2/workspace/test_with_batch.py \
    --model_path "$MODEL_PATH" \
    --test_file "$TEST_FILE" \
    --saved_dir "$SAVED_DIR" > /data5/home/xiechenyu2023/project/ChartQA/Qwen2/workspace/logs/test_qwen2.5_ChartQA_CoT_checkpoint-23000_v2.log 2>&1 &