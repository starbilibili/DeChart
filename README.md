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
