import os
import argparse
import torch
from PIL import Image
from termcolor import colored
from dcag.scheduler import FlowMatchEulerDiscreteScheduler
from dcag.pipeline import QwenImageEditPipeline
from dcag.models import QwenImageTransformer2DModel
from dcag.utils import seed_everything

seed_everything(42)

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="Qwen/Qwen-Image-Edit")
parser.add_argument("--image_path", type=str, required=True)
parser.add_argument("--edit_prompt", type=str, required=True)
parser.add_argument("--out_path", type=str, default='./results')
parser.add_argument("--cond_b", type=float, default=1.0, help="K-channel bias scale")
parser.add_argument("--cond_delta", type=float, default=1.10, help="K-channel delta scale")
parser.add_argument("--v_bias", type=float, default=1.0, help="V-channel bias scale (1.0=identity)")
parser.add_argument("--v_delta", type=float, default=1.0, help="V-channel delta scale (1.0=identity)")
args = parser.parse_args()

scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
    os.path.join(args.model_path, "scheduler"),
    torch_dtype=torch.bfloat16,
)

transformer = QwenImageTransformer2DModel.from_pretrained(
    os.path.join(args.model_path, "transformer"),
    torch_dtype=torch.bfloat16,
)

pipeline = QwenImageEditPipeline.from_pretrained(
    args.model_path, torch_dtype=torch.bfloat16,
    scheduler=scheduler, transformer=transformer,
)
print("pipeline loaded")
pipeline.to(torch.bfloat16)
pipeline.to("cuda")
pipeline.set_progress_bar_config(disable=None)

out_path = args.out_path
os.makedirs(out_path, exist_ok=True)
print(colored(out_path, color="green"))

input_image = Image.open(args.image_path).convert('RGB').resize((1024, 1024))

# DCAG: 5-tuple = (len, k_bias, k_delta, v_bias, v_delta)
grag_scale = [((512, 1.0, 1.0), (4096, args.cond_b, args.cond_delta, args.v_bias, args.v_delta))] * 60

inputs = {
    "image": input_image,
    "prompt": args.edit_prompt,
    "generator": torch.manual_seed(42),
    "true_cfg_scale": 4.0,
    "negative_prompt": " ",
    "num_inference_steps": 24,
    "return_dict": False,
    "grag_scale": grag_scale,
}

with torch.inference_mode():
    output = pipeline(**inputs)
    image, x0_images, saved_outputs = output

fname = f"{os.path.basename(args.image_path)}_kb{args.cond_b}_kd{args.cond_delta}_vb{args.v_bias}_vd{args.v_delta}.jpg"
image[0].save(os.path.join(out_path, fname))
print(colored(f"Saved: {fname}", color="green"))
