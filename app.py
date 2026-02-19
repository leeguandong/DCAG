"""
DCAG Gradio Demo — Dual-Channel Attention Guidance for Image Editing.

Loads the Qwen-Image-Edit model once and provides a web UI for
interactive editing with K-channel and V-channel control.
"""

import os
from typing import Optional

import gradio as gr
import torch
from PIL import Image
from huggingface_hub import snapshot_download
from hacked_models.scheduler import FlowMatchEulerDiscreteScheduler
from hacked_models.pipeline import QwenImageEditPipeline
from hacked_models.models import QwenImageTransformer2DModel
from hacked_models.utils import seed_everything

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_DTYPE = torch.bfloat16 if _DEVICE == "cuda" else torch.float32
_PIPELINE: Optional[QwenImageEditPipeline] = None
_LOADED_MODEL_PATH: Optional[str] = None


def _load_pipeline(model_path: str) -> QwenImageEditPipeline:
    """Load (or reuse) the pipeline for the given model_path."""
    global _PIPELINE, _LOADED_MODEL_PATH
    if _PIPELINE is not None and _LOADED_MODEL_PATH == model_path:
        return _PIPELINE

    seed_everything(42)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        os.path.join(model_path, "scheduler"), torch_dtype=_DTYPE)
    transformer = QwenImageTransformer2DModel.from_pretrained(
        os.path.join(model_path, "transformer"), torch_dtype=_DTYPE)
    pipe = QwenImageEditPipeline.from_pretrained(
        model_path, torch_dtype=_DTYPE, scheduler=scheduler, transformer=transformer)
    pipe.set_progress_bar_config(disable=None)
    pipe.to(_DTYPE)
    pipe.to(_DEVICE)

    _PIPELINE = pipe
    _LOADED_MODEL_PATH = model_path
    return pipe


def predict(image, edit_prompt, cond_b, cond_delta, v_bias, v_delta):
    if image is None or not edit_prompt:
        return None

    input_image = image.convert("RGB").resize((1024, 1024))
    # DCAG: 5-tuple = (len, k_bias, k_delta, v_bias, v_delta)
    grag_scale = [((512, 1.0, 1.0), (4096, cond_b, cond_delta, v_bias, v_delta))] * 60

    inputs = {
        "image": input_image,
        "prompt": edit_prompt,
        "generator": torch.manual_seed(42),
        "true_cfg_scale": 4.0,
        "negative_prompt": " ",
        "num_inference_steps": 24,
        "return_dict": False,
        "grag_scale": grag_scale,
    }

    with torch.inference_mode():
        image_batch, x0_images, saved_outputs = pipe(**inputs)

    return image_batch[0]


model_dir = "Qwen-Image-Edit"
repo_id = "Qwen/Qwen-Image-Edit"

if not os.path.exists(model_dir) or not os.listdir(model_dir):
    snapshot_download(repo_id=repo_id, local_dir=model_dir, local_dir_use_symlinks=False)
    print(f"Model downloaded to {model_dir}")
else:
    print(f"Model already exists at {model_dir}")

pipe = _load_pipeline(model_dir)

with gr.Blocks(title="DCAG — Dual-Channel Attention Guidance") as demo:
    gr.Markdown("# DCAG — Dual-Channel Attention Guidance\nUpload an image, enter your edit instruction, and control K/V channels.")

    with gr.Row():
        in_image = gr.Image(label="Input Image", type="pil")
        out_image = gr.Image(label="Edited Output", type="pil")

    edit_prompt = gr.Textbox(label="Edit Instruction", placeholder="e.g., Change the color of the rose from red to blue")
    with gr.Row():
        cond_b = gr.Slider(label="K bias (cond_b)", minimum=0.8, maximum=2.0, value=1.0, step=0.01)
        cond_delta = gr.Slider(label="K delta (cond_delta)", minimum=0.8, maximum=2.0, value=1.10, step=0.01)
    with gr.Row():
        v_bias_slider = gr.Slider(label="V bias (v_bias)", minimum=0.8, maximum=2.0, value=1.0, step=0.01)
        v_delta_slider = gr.Slider(label="V delta (v_delta)", minimum=0.8, maximum=2.0, value=1.0, step=0.01)

    run_btn = gr.Button("Run Edit")

    run_btn.click(
        fn=predict,
        inputs=[in_image, edit_prompt, cond_b, cond_delta, v_bias_slider, v_delta_slider],
        outputs=[out_image],
        api_name="run_edit",
    )

    gr.Markdown(
        """
**Notes**
- K-channel controls attention routing (coarse). V-channel controls feature aggregation (fine).
- Recommended: cond_delta=1.10, v_delta=1.0~1.15 for best fidelity-quality trade-off.
- Uses fixed seed=42, 24 inference steps, CFG scale=4.0.
        """
    )

if __name__ == "__main__":
    demo.queue().launch(share=True)
