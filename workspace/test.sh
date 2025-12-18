# 设置 CUDA 设备
export CUDA_VISIBLE_DEVICES=0

# 定义变量
MODEL_ROOT="/lustre/home/xiechenyu2023/saved_model/KPDetect/v1"  # 模型根目录
OUTPUT_ROOT="/data5/home/xiechenyu2023/project/ChartQA/DeChart_workplace/KPDetection_module/data/output"  # 输出结果根目录

# 单模型测试
# MODEL_NAME=Qwen2-VL-7B-Instruct
MODEL_NAME=Qwen_KPDetect_1018_checkpoint-630
# 构建完整的模型路径
MODEL_PATH="$MODEL_ROOT/$MODEL_NAME"
# 构建输出目录名称
SAVED_DIR="$OUTPUT_ROOT/${MODEL_NAME}"

# 打印当前处理的模型信息
echo "Processing model: $MODEL_NAME"
echo "Model path: $MODEL_PATH"
echo "Output directory: $SAVED_DIR"

# 执行 test.py 脚本
python /data5/home/xiechenyu2023/project/ChartQA/Qwen2/workspace/test.py \
    --model_path "$MODEL_PATH" \
    --saved_dir "$SAVED_DIR" > /data5/home/xiechenyu2023/project/ChartQA/Qwen2/workspace/logs/test_Chart2table_Dechart_1006_checkpoint-12000.log 2>&1 &