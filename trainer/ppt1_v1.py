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

class PowerPaint:
    def __init__(self, args):
        self.args = args
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            torch_dtype=args.weight_dtype,
            local_files_only=args.local_files_only,
        )

        