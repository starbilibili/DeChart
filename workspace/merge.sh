export CUDA_VISIBLE_DEVICES=3

CHECKPOINT=1500
# MODEL_NAME_OR_PATH=/data5/home/xiechenyu2023/project/ChartQA/Qwen2/pretrained_model/Qwen2-VL-2B-Instruct
MODEL_NAME_OR_PATH=/lustre/home/xiechenyu2023/saved_model/qwen2/Qwen2.5-VL-7B-Instruct
ADAPTER_NAME_OR_PATH=/lustre/home/xiechenyu2023/saved_model/saves/qwen2.5_vl-7b/KPDetect/v3/checkpoint-${CHECKPOINT}
EXPORT_DIR=/lustre/home/xiechenyu2023/saved_model/KPDetect/v3/Qwen_KPDetect_1018_checkpoint-${CHECKPOINT}

CONFIG_FILE="examples/merge_lora/qwen2vl_lora_sft.yaml"
# 备份原始配置文件
cp "$CONFIG_FILE" "${CONFIG_FILE}.bak"

# 直接在源配置文件中替换参数
sed -i -e "s|model_name_or_path: .*|model_name_or_path: $MODEL_NAME_OR_PATH|" \
       -e "s|adapter_name_or_path: .*|adapter_name_or_path: $ADAPTER_NAME_OR_PATH|" \
       -e "s|export_dir: .*|export_dir: $EXPORT_DIR|" \
       "$CONFIG_FILE" || { echo "Failed to update config file"; exit 1; }
# 打印更新后的配置文件内容
echo "Updated config file:"
cat "$CONFIG_FILE"

# 调用 llamafactory-cli export 命令
llamafactory-cli export "$CONFIG_FILE" || { echo "Export failed"; exit 1; }

# 恢复原始配置文件
mv "${CONFIG_FILE}.bak" "$CONFIG_FILE"