# DeChart

## Model Weights
**Download:** [seven-night/DeChart](https://huggingface.co/seven-night/DeChart)

Our model is fine-tuned from **Qwen2.5-VL-7B**, incorporating:
- OCR-enhanced text-visual fusion
- Two-stage Chain-of-Thought reasoning
- Stage-adaptive training strategy

### Key Capabilities:
- ✔ Accurate extraction of chart data elements
- ✔ Structured table generation from complex charts

  
## Dataset
**Download:** https://huggingface.co/datasets/seven-night/DeChart

**DeChart** is our newly proposed challenging benchmark for Chart-to-Table tasks, featuring:

- 29,450 meticulously annotated charts (Bar/Pie/Line)

- Enhanced diversity: Includes charts without explicit numerical labels

- Dual annotation sets:

  - Complete table structure annotations

  - Fine-grained text element labels (title/axis/legend etc.)

- 10%+ performance gap vs. existing benchmarks (ChartQA)

**Designed to evaluate models' capabilities in:**
  ▸ Visual-textual alignment
  ▸ Implicit numerical reasoning
  ▸ Structural relationship preservation


## Usage Guide

To reproduce our results or fine-tune / evaluate your own variant, please follow the instructions below.

### 🔍 Inference

1. **Download** the model weights and dataset:
   - Model: [seven-night/DeChart](https://huggingface.co/seven-night/DeChart)
   - Dataset: [seven-night/DeChart (dataset)](https://huggingface.co/datasets/seven-night/DeChart)

2. **Configure paths**:
   - In `test.sh`, set:
     ```bash
     MODEL_PATH=/path/to/your/model
     OUTPUT_DIR=/path/to/save/predictions
     ```
   - In `test.py`, specify the test set path:
     ```python
     test_data_path = "/path/to/DeChart/test"
     ```

3. **Run inference**:
   ```bash
   bash test.sh
   ```

This will generate structured table predictions in JSON or CSV format under `OUTPUT_DIR`.

### 2. Evaluation
#### Step 1: Configuration
Modify `evaluation.py` to specify:
- Path to the folder containing model prediction results (`pred_dir`)
- Path to the folder containing ground truth annotations (`gt_dir`)

#### Step 2: Run Evaluation
Execute the evaluation script to get quantitative results:
```bash
python evaluation.py
```

#### Notes on Evaluation Metrics
- Modify the `version` parameter in `evaluation.py` to switch between different evaluation metric versions.
- For detailed differences between the two metric versions, please refer to our paper.

### 3. Training
Our training process follows the Llama-Factory framework. Below are the key hyperparameters used in our paper experiments for reference:

#### Basic Configuration
```yaml
### model
model_name_or_path: /lustre/home/xiechenyu2023/saved_model/qwen2/Qwen2.5-VL-7B-Instruct
image_max_pixels: 524176
video_max_pixels: 16384
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: lora
lora_rank: 8
lora_target: all   # ['down_proj', 'gate_proj', 'o_proj', 'v_proj', 'q_proj', 'up_proj', 'k_proj']

### dataset
dataset: keypoint_detect_with_axis # video: mllm_video_demo
template: qwen2_vl
cutoff_len: 2048
max_samples: 30000
overwrite_cache: true
preprocessing_num_workers: 16
dataloader_num_workers: 4

### output
output_dir: /lustre/home/xiechenyu2023/saved_model/saves/qwen2.5_vl-7b/KPDetect/v3
logging_steps: 10
save_steps: 500
plot_loss: true
overwrite_output_dir: true
save_only_model: false

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 1
learning_rate: 1.0e-6
num_train_epochs: 100
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000
resume_from_checkpoint: null

### eval
val_size: 0.1
per_device_eval_batch_size: 1
eval_strategy: steps
eval_steps: 500
```
### Training Execution
Follow the official Llama-Factory documentation to set up the environment and run the training script with the above configuration file.

