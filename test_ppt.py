import argparse
import os
import random

from omegaconf import OmegaConf
import cv2
import numpy as np
import torch
from accelerate.utils import set_seed
from PIL import Image, ImageFilter
from transformers import CLIPTextModel
from safetensors.torch import load_model

from powerpaint.models import UNet2DConditionModel
from powerpaint.pipelines import StableDiffusionInpaintPipeline

torch.set_grad_enabled(False)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


class PowerPaint:
    def __init__(self, args):
        self.args = args
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
          #  torch_dtype=args.weight_dtype,
            local_files_only=True
        )
        #loading pre-trained weights
        load_model(self.pipe.unet, os.path.join(args.ppt1_model_path, "unet/unet.safetensors"), strict=False)
        load_model(self.pipe.text_encoder, os.path.join(args.ppt1_model_path, "text_encoder/text_encoder.safetensors"),strict=False )

        self.pipe.add_tokens(
            placeholder_tokens=["P_obj", "P_ctxt", "P_shape"],
            initializer_tokens=["a", "a", "a"],
            num_vectors_per_token=10,
            initialize_parameters=False,
        )
      #  self.pipe.enable_model_cpu_offload()
        print("Tokenizer vocab:", len(self.pipe.tokenizer))
        print("Text encoder vocab:", self.pipe.text_encoder.get_input_embeddings().weight.shape[0])
        self.pipe = self.pipe.to("cuda")

    def validation(self):
        torch.set_grad_enabled(False)
       # self.pipe.eval()

        for case in self.args.validation_data.cases:
            validation_prompts = case.prompt
            validation_image = Image.open(os.path.join(self.args.validation_data.data_root, case.image)).convert("RGB")
            validation_mask = Image.open(os.path.join(self.args.validation_data.data_root, case.mask)).convert('L')

            image_grid = Image.new(
                "RGB",
                (validation_image.size[0] * (1 + len(validation_prompts)), validation_image.size[1]),
                (255, 255, 255),
            )
            image_grid.paste(validation_image, (0, 0))

            for i, p in enumerate(validation_prompts):
                image = self.pipe(
                    promptA=p.promptA,
                    promptB=p.promptB,
                    negative_promptA=p.get("negative_promptA", None),
                    negative_promptB=p.get("negative_promptB", None),
                    tradeoff=p.tradeoff,
                    image=validation_image,
                    mask=validation_mask,
                    num_inference_steps=45,
                ).images[0]
                image_grid.paste(image, (validation_image.size[0] * (i + 1), 0))
            save_path = os.path.join(
                self.args.output_dir,
                f"{case.name}_{os.path.basename(case.image)}"
            )
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            image_grid.save(save_path)

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="yaml for configuration",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=False,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="runs/ppt1_sd15",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    args = parser.parse_args()
    # use omegaconf to manage configurations
    if args.config is not None:
        config = OmegaConf.load(args.config)
        for k, v in config.items():
            args.__dict__[k] = v
    return args

def main():
    args = parse_args()
    model = PowerPaint(args)
    model.validation()

if __name__ == '__main__':
    set_seed(42)
    main()



