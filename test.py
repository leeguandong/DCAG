import os
import json
import argparse
import torch
import tqdm
from PIL import Image
from termcolor import colored
from hacked_models.scheduler import FlowMatchEulerDiscreteScheduler
from hacked_models.pipeline import QwenImageEditPipeline
from hacked_models.models import QwenImageTransformer2DModel
from hacked_models.utils import seed_everything

seed_everything(42)

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="Qwen/Qwen-Image-Edit")
parser.add_argument("--data_root", type=str, required=True)
parser.add_argument("--out_path", type=str, required=True)
parser.add_argument("--name", type=str, required=True)
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

DATA_ROOT = args.data_root
data_root = '/'.join(DATA_ROOT.split('/')[:-1])
out_path = os.path.join(args.out_path, DATA_ROOT.split('/')[-2], args.name)
os.makedirs(out_path, exist_ok=True)

with open(DATA_ROOT, 'r') as f:
    test_pairs = json.load(f)
print(colored(out_path, color="green"))

for name, pair in tqdm.tqdm(list(test_pairs.items())):
    img_folder = '/'.join(pair["image_path"].split('/')[:-1])
    img_name = pair["image_path"].split('/')[-1]
    editing_instruction = pair["editing_instruction"]

    input_image = Image.open(os.path.join(data_root, img_folder, img_name)).convert('RGB')
    os.makedirs(os.path.join(out_path, img_folder), exist_ok=True)
    if os.path.exists(os.path.join(out_path, img_folder, img_name)):
        print('Already generated.')
        continue

    # DCAG: 5-tuple = (len, k_bias, k_delta, v_bias, v_delta)
    grag_scale = [((512, 1.0, 1.0), (4096, args.cond_b, args.cond_delta, args.v_bias, args.v_delta))] * 60

    inputs = {
        "image": input_image,
        "prompt": editing_instruction,
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

    image[0].save(os.path.join(out_path, img_folder, img_name))
