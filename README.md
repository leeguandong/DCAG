# Dual-Channel Attention Guidance (DCAG) for Image Editing

[![arXiv](https://img.shields.io/badge/arXiv-Paper-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/)

> **Dual-Channel Attention Guidance for Training-Free Image Editing Control in Diffusion Transformers**
> Guandong Li, Mengxia Ye

DCAG is a training-free mechanism for fine-grained control over image editing intensity in Diffusion-in-Transformer (DiT) models. It simultaneously manipulates both Key (attention routing) and Value (feature aggregation) channels in the transformer's multi-modal attention, providing a 2D parameter space for precise editing-fidelity control.

---

## Overview

In DiT-based image editing models, the attention output is determined by two orthogonal channels:
- **K-channel (δ_k)**: Controls *where to look* — attention routing via softmax nonlinearity (coarse control)
- **V-channel (δ_v)**: Controls *what to see* — feature aggregation via linear combination (fine control)

DCAG decomposes both Key and Value token embeddings into bias + delta components, then rescales the deltas to control editing strength. GRAG (K-only) is a special case of DCAG where δ_v = 1.0.

---

## Repository Structure

```
DCAG/
├── dcag/                  # Core DCAG implementation (modified HuggingFace Diffusers)
│   ├── models.py          # DCAG injection: K+V bias-delta reweighting in attention
│   ├── pipeline.py        # QwenImageEditPipeline with DCAG support
│   ├── attention.py       # Dual-stream attention processor with RoPE
│   ├── scheduler.py       # FlowMatchEulerDiscreteScheduler
│   └── utils.py           # Utilities
├── inference.py           # Single-image editing script
├── test.py                # PIE benchmark batch evaluation
├── app.py                 # Gradio web UI
├── requirements.txt
└── assets/                # Sample images
```

---

## Installation

```bash
git clone https://github.com/leeguandong/DCAG.git
cd DCAG
conda create -n dcag python=3.10 -y
conda activate dcag
pip install -r requirements.txt
```

Models auto-download from HuggingFace (`Qwen/Qwen-Image-Edit`) on first run.

---

## Usage

### Single image editing

```bash
python inference.py \
    --model_path Qwen/Qwen-Image-Edit \
    --image_path assets/sample.jpg \
    --edit_prompt "Change the color of the rose from red to blue" \
    --cond_delta 1.10 --v_delta 1.15
```

### PIE benchmark batch test

```bash
python test.py \
    --model_path Qwen/Qwen-Image-Edit \
    --data_root DATA_PATH/PIE/mapping_file.json \
    --out_path ./results --name dcag_test \
    --cond_delta 1.10 --v_delta 1.15
```

### Gradio web UI

```bash
python app.py
```

---

## Parameters

| Parameter | CLI flag | Range | Default | Description |
|-----------|----------|-------|---------|-------------|
| K-bias | `--cond_b` | 0.8–1.7 | 1.0 | Key bias scale |
| K-delta | `--cond_delta` | 0.8–1.7 | 1.10 | Key delta scale (attention routing) |
| V-bias | `--v_bias` | 0.8–1.7 | 1.0 | Value bias scale |
| V-delta | `--v_delta` | 0.8–1.7 | 1.0 | Value delta scale (feature aggregation) |

**Recommended configurations:**
- K-only (GRAG equivalent): `--cond_delta 1.10`
- DCAG best fidelity: `--cond_delta 1.10 --v_delta 1.15`
- Strong preservation: `--cond_delta 1.20`

---

## GRAG Scale Format

DCAG uses a 5-tuple format extending GRAG's 3-tuple:

```python
# 3-tuple (GRAG, K-only): (seq_len, k_bias, k_delta)
# 5-tuple (DCAG, K+V):    (seq_len, k_bias, k_delta, v_bias, v_delta)
grag_scale = [((512, 1.0, 1.0), (4096, 1.0, 1.10, 1.0, 1.15))] * 60
```

---

## Citation

If you find this work useful, please cite:

```bibtex
@article{li2025dcag,
  title={Dual-Channel Attention Guidance for Training-Free Image Editing Control in Diffusion Transformers},
  author={Li, Guandong and Ye, Mengxia},
  year={2025}
}
```

## Acknowledgements

This work builds upon [GRAG (Group Relative Attention Guidance)](https://arxiv.org/abs/2510.24657) and uses the [Qwen-Image-Edit](https://huggingface.co/Qwen/Qwen-Image-Edit) model. We thank the authors for their contributions.
