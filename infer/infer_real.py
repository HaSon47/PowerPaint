import argparse
import os
import random
import json

import cv2
import gradio as gr
import numpy as np
import torch
from controlnet_aux import HEDdetector, OpenposeDetector
from PIL import Image, ImageFilter, ImageDraw
from tqdm import tqdm
from safetensors.torch import load_model
from transformers import CLIPTextModel, DPTFeatureExtractor, DPTForDepthEstimation

from diffusers import UniPCMultistepScheduler
from diffusers.pipelines.controlnet.pipeline_controlnet import ControlNetModel
from powerpaint.models.BrushNet_CA import BrushNetModel
from powerpaint.models.unet_2d_condition import UNet2DConditionModel
from powerpaint.pipelines.pipeline_PowerPaint import StableDiffusionInpaintPipeline as Pipeline
from powerpaint.pipelines.pipeline_PowerPaint_Brushnet_CA import StableDiffusionPowerPaintBrushNetPipeline
from powerpaint.pipelines.pipeline_PowerPaint_ControlNet import (
    StableDiffusionControlNetInpaintPipeline as controlnetPipeline,
)
from powerpaint.utils.utils import TokenizerWrapper, add_tokens


torch.set_grad_enabled(False)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def add_task(prompt, negative_prompt, control_type, version):
    pos_prefix = neg_prefix = ""
    
    if control_type == "ctxt-guided":
        if version == 'ppt-v1':
            pos_prefix = prompt
            neg_prefix = negative_prompt + ", worst quality, low quality, normal quality, bad quality, blurry "
        promptA = pos_prefix + " P_obj"
        promptB = pos_prefix + " P_ctxt"
        negative_promptA = neg_prefix + " P_obj"
        negative_promptB = neg_prefix + " P_ctxt"
    else:
        if version == "ppt-v1":
            pos_prefix = prompt
            neg_prefix = negative_prompt + ", worst quality, low quality, normal quality, bad quality, blurry "
        promptA = pos_prefix + " P_obj"
        promptB = pos_prefix + " P_obj"
        negative_promptA = neg_prefix + " P_obj"
        negative_promptB = neg_prefix + " P_obj"

    return promptA, promptB, negative_promptA, negative_promptB


class PowerPaint:
    def __init__(self, weight_dtype, checkpoint_dir, local_files_only, version):
        self.version = version
        self.checkpoint_dir = checkpoint_dir
        self.local_file_only = local_files_only

        # initialize powerpaint pipeline
        if version == 'ppt-v1':
            self.pipe = Pipeline.from_pretrained(
                '/workdir/radish/hachi/checkpoints/stable-diffusion-inpainting',
                torch_dtype=weight_dtype,
                local_files_only=True
            )
            self.pipe.tokenizer = TokenizerWrapper(
                from_pretrained="/workdir/radish/hachi/checkpoints/stable-diffusion-v1-5",
                subfolder='tokenizer',
                revision=None,
                local_files_only=True,
            )

            # add learned task tokens into the tokenizer
            add_tokens(
                tokenizer=self.pipe.tokenizer,
                text_encoder=self.pipe.text_encoder,
                placeholder_tokens=["P_ctxt", "P_shape", "P_obj"],
                initialize_tokens=["a", "a", "a"],
                num_vectors_per_token=10,
            )
            # loading pre-trained weights
            load_model(self.pipe.unet, os.path.join(checkpoint_dir, "unet/unet.safetensors"), strict=False)
            load_model(self.pipe.text_encoder, os.path.join(checkpoint_dir, "text_encoder/text_encoder.safetensors"),strict=False )
            self.pipe = self.pipe.to("cuda")

            # initialize controlnet-related models
            self.depth_estimator = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas").to("cuda")
            self.feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-hybrid-midas")
            self.openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
            self.hed = HEDdetector.from_pretrained("lllyasviel/ControlNet")

            # base_control = ControlNetModel.from_pretrained(
            #     "lllyasviel/sd-controlnet-canny", torch_dtype=weight_dtype, local_files_only=local_files_only
            # )
            # self.control_pipe = controlnetPipeline(
            #     self.pipe.vae,
            #     self.pipe.text_encoder,
            #     self.pipe.tokenizer,
            #     self.pipe.unet,
            #     base_control,
            #     self.pipe.scheduler,
            #     None,
            #     None,
            #     False,
            # )
            # self.control_pipe = self.control_pipe.to("cuda")

            # self.current_control = "canny"
            # # controlnet_conditioning_scale = 0.8

        else:
            # brushnet-based version
            unet = UNet2DConditionModel.from_pretrained(
                # "sd-legacy/stable-diffusion-v1-5",
                '/workdir/radish/hachi/checkpoints/stable-diffusion-v1-5',
                subfolder="unet",
                revision=None,
                torch_dtype=weight_dtype,
                local_files_only=local_files_only,
            )
            text_encoder_brushnet = CLIPTextModel.from_pretrained(
                # "sd-legacy/stable-diffusion-v1-5",
                '/workdir/radish/hachi/checkpoints/stable-diffusion-v1-5',
                subfolder="text_encoder",
                revision=None,
                torch_dtype=weight_dtype,
                local_files_only=local_files_only,
            )
            brushnet = BrushNetModel.from_unet(unet)
            base_model_path = os.path.join(checkpoint_dir, "realisticVisionV60B1_v51VAE")
            self.pipe = StableDiffusionPowerPaintBrushNetPipeline.from_pretrained(
                base_model_path,
                brushnet=brushnet,
                text_encoder_brushnet=text_encoder_brushnet,
                torch_dtype=weight_dtype,
                low_cpu_mem_usage=False,
                safety_checker=None,
            )
            self.pipe.unet = UNet2DConditionModel.from_pretrained(
                base_model_path,
                subfolder="unet",
                revision=None,
                torch_dtype=weight_dtype,
                local_files_only=local_files_only,
            )
            self.pipe.tokenizer = TokenizerWrapper(
                from_pretrained=base_model_path,
                subfolder="tokenizer",
                revision=None,
                torch_type=weight_dtype,
                local_files_only=local_files_only,
            )

            # add learned task tokens into the tokenizer
            add_tokens(
                tokenizer=self.pipe.tokenizer,
                text_encoder=self.pipe.text_encoder_brushnet,
                placeholder_tokens=["P_ctxt", "P_shape", "P_obj"],
                initialize_tokens=["a", "a", "a"],
                num_vectors_per_token=10,
            )
            load_model(
                self.pipe.brushnet,
                os.path.join(checkpoint_dir, "PowerPaint_Brushnet/diffusion_pytorch_model.safetensors"),
                strict=False
            )

            self.pipe.text_encoder_brushnet.load_state_dict(
                torch.load(os.path.join(checkpoint_dir, "PowerPaint_Brushnet/pytorch_model.bin")), strict=False
            )

            self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)

            #self.pipe.enable_model_cpu_offload()
            self.pipe = self.pipe.to("cuda")

    def get_depth_map(self, image):
        image = self.feature_extractor(images=image, return_tensors="pt").pixel_values.to("cuda")
        with torch.no_grad(), torch.autocast("cuda"):
            depth_map = self.depth_estimator(image).predicted_depth

        depth_map = torch.nn.functional.interpolate(
            depth_map.unsqueeze(1),
            size=(1024, 1024),
            mode="bicubic",
            align_corners=False,
        )
        depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
        depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
        depth_map = (depth_map - depth_min) / (depth_max - depth_min)
        image = torch.cat([depth_map] * 3, dim=1)

        image = image.permute(0, 2, 3, 1).cpu().numpy()[0]
        image = Image.fromarray((image * 255.0).clip(0, 255).astype(np.uint8))
        return image

    def predict(
        self,
        input_image,
        prompt,
        fitting_degree,
        ddim_steps,
        scale,
        seed,
        negative_prompt,
        task,
        vertical_expansion_ratio,
        horizontal_expansion_ratio,
    ):
        size1, size2 = input_image["image"].convert("RGB").size

        if task != "image-outpainting":
            if size1 < size2:
                input_image["image"] = input_image["image"].convert("RGB").resize((640, int(size2 / size1 * 640)))
            else:
                input_image["image"] = input_image["image"].convert("RGB").resize((int(size1 / size2 * 640), 640))
        else:
            if size1 < size2:
                input_image["image"] = input_image["image"].convert("RGB").resize((512, int(size2 / size1 * 512)))
            else:
                input_image["image"] = input_image["image"].convert("RGB").resize((int(size1 / size2 * 512), 512))

        if vertical_expansion_ratio is not None and horizontal_expansion_ratio is not None:
            o_W, o_H = input_image["image"].convert("RGB").size
            c_W = int(horizontal_expansion_ratio * o_W)
            c_H = int(vertical_expansion_ratio * o_H)

            expand_img = np.ones((c_H, c_W, 3), dtype=np.uint8) * 127
            original_img = np.array(input_image["image"])
            expand_img[
                int((c_H - o_H) / 2.0) : int((c_H - o_H) / 2.0) + o_H,
                int((c_W - o_W) / 2.0) : int((c_W - o_W) / 2.0) + o_W,
                :,
            ] = original_img

            blurry_gap = 10

            expand_mask = np.ones((c_H, c_W, 3), dtype=np.uint8) * 255
            if vertical_expansion_ratio == 1 and horizontal_expansion_ratio != 1:
                expand_mask[
                    int((c_H - o_H) / 2.0) : int((c_H - o_H) / 2.0) + o_H,
                    int((c_W - o_W) / 2.0) + blurry_gap : int((c_W - o_W) / 2.0) + o_W - blurry_gap,
                    :,
                ] = 0
            elif vertical_expansion_ratio != 1 and horizontal_expansion_ratio != 1:
                expand_mask[
                    int((c_H - o_H) / 2.0) + blurry_gap : int((c_H - o_H) / 2.0) + o_H - blurry_gap,
                    int((c_W - o_W) / 2.0) + blurry_gap : int((c_W - o_W) / 2.0) + o_W - blurry_gap,
                    :,
                ] = 0
            elif vertical_expansion_ratio != 1 and horizontal_expansion_ratio == 1:
                expand_mask[
                    int((c_H - o_H) / 2.0) + blurry_gap : int((c_H - o_H) / 2.0) + o_H - blurry_gap,
                    int((c_W - o_W) / 2.0) : int((c_W - o_W) / 2.0) + o_W,
                    :,
                ] = 0

            input_image["image"] = Image.fromarray(expand_img)
            input_image["mask"] = Image.fromarray(expand_mask)

        promptA, promptB, negative_promptA, negative_promptB = add_task(prompt, negative_prompt, task, self.version)
        print(promptA, promptB, negative_promptA, negative_promptB)

        img = np.array(input_image["image"].convert("RGB"))
        W = int(np.shape(img)[0] - np.shape(img)[0] % 8)
        H = int(np.shape(img)[1] - np.shape(img)[1] % 8)
        input_image["image"] = input_image["image"].resize((H, W))
        input_image["mask"] = input_image["mask"].resize((H, W))
        set_seed(seed)
        if self.version == "ppt-v1":
            # for sd-inpainting based method
            result = self.pipe(
                promptA=promptA,
                promptB=promptB,
                tradoff=fitting_degree,
                tradoff_nag=fitting_degree,
                negative_promptA=negative_promptA,
                negative_promptB=negative_promptB,
                image=input_image["image"].convert("RGB"),
                mask=input_image["mask"].convert("RGB"),
                width=H,
                height=W,
                guidance_scale=scale,
                num_inference_steps=ddim_steps,
            ).images[0]

        else:
            # for brushnet-based method
            np_inpimg = np.array(input_image["image"])
            np_inmask = np.array(input_image["mask"]) / 255.0
            np_inpimg = np_inpimg * (1 - np_inmask)
            input_image["image"] = Image.fromarray(np_inpimg.astype(np.uint8)).convert("RGB")
            result = self.pipe(
                promptA=promptA,
                promptB=promptB,
                promptU=prompt,
                tradoff=fitting_degree,
                tradoff_nag=fitting_degree,
                image=input_image["image"].convert("RGB"),
                mask=input_image["mask"].convert("RGB"),
                num_inference_steps=ddim_steps,
                generator=torch.Generator("cuda").manual_seed(seed),
                brushnet_conditioning_scale=1.0,
                negative_promptA=negative_promptA,
                negative_promptB=negative_promptB,
                negative_promptU=negative_prompt,
                guidance_scale=scale,
                width=H,
                height=W,
            ).images[0]

        # Lấy bbox gốc (theo định dạng [x_min, y_min, x_max, y_max])
        x_min, y_min, x_max, y_max = input_image["bbox"]

        # Tính tỉ lệ resize (ảnh gốc → ảnh hiện tại)
        # Lưu ý: size ban đầu (size1, size2) lấy ở đầu hàm
        new_W, new_H = input_image["image"].size  # sau khi resize
        scale_x = new_W / size1
        scale_y = new_H / size2

        # Scale bbox theo tỉ lệ
        x_min = int(x_min * scale_x)
        x_max = int(x_max * scale_x)
        y_min = int(y_min * scale_y)
        y_max = int(y_max * scale_y)

        # Tạo bản copy của ảnh kết quả và ảnh input để vẽ viền
        result_m = result.copy()

        # Vẽ viền bbox màu xanh lá (RGB = (0, 255, 0))
        draw_result = ImageDraw.Draw(result_m)
        draw_result.rectangle([x_min, y_min, x_max, y_max], outline=(0, 255, 0), width=3)

        return result_m

    def infer(
        self,
        input_image,
        text_guided_prompt,
        text_guided_negative_prompt,
        shape_guided_prompt,
        shape_guided_negative_prompt,
        fitting_degree,
        ddim_steps,
        scale,
        seed,
        task,
    ):
        
        if task == "text-guided":
            prompt = text_guided_prompt
            negative_prompt = text_guided_negative_prompt
        elif task == "ctxt-guided":
            prompt = shape_guided_prompt
            negative_prompt = shape_guided_negative_prompt


        return self.predict(
            input_image, prompt, fitting_degree, ddim_steps, scale, seed, negative_prompt, task, None, None
        )
    
def create_mask_from_bbox(img_size, bbox):
    """
    Tạo mask nhị phân từ bbox.

    Args:
        img_size: tuple (width, height) của ảnh gốc
        bbox: tuple (x_min, y_min, x_max, y_max)

    Returns:
        PIL.Image mask
    """
    # Tạo ảnh mask toàn đen
    mask = Image.new("L", img_size, 0)
    draw = ImageDraw.Draw(mask)

    # Vẽ hình chữ nhật trắng theo bbox
    draw.rectangle(bbox, fill=255)

    return mask
    

def get_inputs(data_folder_path):
    print('get input')
    inputs = {}
    img_folder_path = os.path.join(data_folder_path, 'input')
    anno_path = os.path.join(data_folder_path, 'annotations.json')
    with open(anno_path, 'r') as f:
        anno = json.load(f)
    for img_file in tqdm(os.listdir(img_folder_path)):
        # get image
        image = Image.open(os.path.join(img_folder_path, img_file))
        img_name = img_file.split('.')[0]

        # get info about image
        ## get class
        class_prompt = anno[img_file]['class_name']

        ## get bbox
        bbox = anno[img_file]['bbox']  # [x_min, y_min, x_max, y_max]

        ## get mask
        mask = create_mask_from_bbox(image.size, bbox) 

        inputs[img_name] = {
           "image": image,
           "mask": mask,
           "class_prompt": class_prompt,
           "bbox": bbox
       } 
    return inputs


def main():
    args = argparse.ArgumentParser()
    args.add_argument("--weight_dtype", type=str, default="float16")
    args.add_argument("--checkpoint_dir", type=str, default='./checkpoints/ppt-v1')
    args.add_argument("--version", type=str, default="ppt-v1")
    args.add_argument(
        "--local_files_only", action="store_true", help="enable it to use cached files without requesting from the hub"
    )
    args.add_argument("--output_path", type=str, required=True)
    args.add_argument("--data_folder_path", type=str, required=True)
    args.add_argument("--task", type=str, required=True)
    args = args.parse_args()

    # initialize the pipeline controller
    weight_dtype = torch.float16 if args.weight_dtype == "float16" else torch.float32
    controller = PowerPaint(weight_dtype, args.checkpoint_dir, args.local_files_only, args.version)

    ## input
    input_images = get_inputs(args.data_folder_path)

    # infer
    #hyperparameter
    fitting_degree = 0.0

    ddim_steps = 50
    scale = 7.5
    seed = 42
    os.makedirs(args.output_path, exist_ok=True)
    print(args.task)
    for img_folder, input_image in tqdm(input_images.items()):
        # run
        result_m = controller.infer(
            input_image,
            text_guided_prompt='a ' + input_image["class_prompt"],
            text_guided_negative_prompt="",
            shape_guided_prompt='a ' + input_image["class_prompt"],
            shape_guided_negative_prompt="",
            fitting_degree=fitting_degree,
            ddim_steps=ddim_steps,
            scale=scale,
            seed=seed,
            task=args.task,
        )

        # save
        result_m.save(f"{args.output_path}/{img_folder}.png")
        # out_path = os.path.join(args.output_path, img_folder)
        # os.makedirs(out_path, exist_ok=True)
        # input_image["image"].save(f"{out_path}/input.png")
        # result.save(f"{out_path}/add.png")
        # mask.save(f"{out_path}/mask.png")
        # result_m.save(f"{out_path}/with_mask.png")
        # input_m.save(f"{out_path}/input_with_mask.png")
        

if __name__ == "__main__":
    main()
