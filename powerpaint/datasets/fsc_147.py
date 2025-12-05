import os
import json
import random

import cv2
import numpy as np
import torch
from accelerate.logging import get_logger
from PIL import Image
from torch.utils.data import IterableDataset
from torchvision import transforms

logger = get_logger(__name__)

def preprocess_img():
    pass

class FSC147_Dataset(IterableDataset):
    def __init__(
            self,
            data_path,
            transforms,
            pipeline,
            task_prompt,
            desc_prefix=False,
            bufsize=None,
            clip_score_threshold=None,
            resolution=None,
            deterministic=False
    ):
        super().__init__()

        # data
        self.data_path = data_path
        # for data sample
        self.bufsize = bufsize
        self.resolution = resolution
        self.epoch = -1
        self.deterministic = deterministic
        self.pipeline = pipeline
        self.task_prompt = task_prompt
        self.desc_prefix = desc_prefix

        # for data filter
        self.clip_score_threshold = clip_score_threshold
        self.transforms = transforms

    def _sample_data(self, data_info):

        # load img
        img = Image.open(data_info["img_path"]).convert('RGB')
        img = preprocess_img(img)
    def sample_data(self):
        buffer = []
        for img_folder in os.listdir(self.data_path):
            # load data
            with open(os.path.join(self.data_path, img_folder, 'annotation.json'), 'r') as f:
                anno = json.load(f)
            mask_bbox = anno['inpainted_bboxes'][0]
            loc_bbox = anno['inpainted_bboxes'][1]
            prompt = anno['class_based_caption']
            img_path = os.path.join(self.data_path, img_folder, 'ground_truth.jpg')

            # make data for task indomain inpainting
            task_type = "indomain_inpainting"
            promptA = self.task_prompt.indomain_inpainting.placeholder_tokens
            promptB = self.task_prompt.indomain_inpainting.placeholder_tokens

            # let see: Null + p_loc
            if random.random()<0.3:
                prompt = ""

            # 10% dropout for unconditional training
            if random.random() < 0.1:
                promptA = promptB = prompt = ""

            data_info = {
                "img_path": img_path,
                "mask_bbox": mask_bbox,
                "loc_bbox": loc_bbox,
                "promptA": promptA,
                "promptB": promptB,
                "prompt": prompt,
                "task_type": task_type
            }
            if self.bufsize is None:
                try:
                    data = self._sample_data(data_info)
                    if data is None:
                        continue
                    else:
                        yield data
                except Exception:
                    logger.info(f"Error in {data_info}")
                    continue

            elif len(buffer) < self.bufsize:
                buffer.append(data_info)

            else:
                select_idx = random.randint(0, self.bufsize - 1)
                selected_data = buffer[select_idx]
                try:
                    data = self._sample_data(selected_data)
                    yield data
                except Exception:
                    logger.info(f"Error in {selected_data}")
                    continue
                buffer[select_idx] = data_info

        for data_info in buffer:
            try:
                yield self._sample_data(data_info)
            except Exception:
                logger.info(f"Error in {data_info}")
                continue

    def __iter__(self):
        for data in self.sample_data():
            yield data

    def __len__(self):
        return 999_999_999
    def __repr__(self):
        return "FSC147_Dataset"
