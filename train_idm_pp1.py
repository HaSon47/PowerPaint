import argparse
import gc
import logging
import math
import os
import shutil
import inspect

import accelerate
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from omegaconf import OmegaConf
from packaging import version
from PIL import Image
from safetensors.torch import load_model
from tqdm.auto import tqdm
from transformers import PretrainedConfig

import diffusers
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel, compute_snr
from diffusers.utils import check_min_version, deprecate, is_wandb_available
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module
from powerpaint.datasets.FSC147 import FSCDataset
from powerpaint.models import UNet2DConditionModel
from powerpaint.pipelines import StableDiffusionInpaintIndomainPipeline
from powerpaint.utils.utils import TokenizerWrapper, add_tokens

# if is_wandb_available():
#     import wandb
#     from dotenv import load_dotenv
#     load_dotenv()
#     wandb.login()

logger = get_logger(__name__, log_level="INFO")

def log_validation(tokenizer, text_encoder, unet, args, accelerator, weight_dtype, step):
    logger.info("Running validation... ")

    pipe = StableDiffusionInpaintIndomainPipeline.from_pretrained(
        args.base_model_path,
        # text_encoder=accelerator.unwrap_model(text_encoder),
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        # unet=accelerator.unwrap_model(unet),
        unet=unet,
        safety_checker=None,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
        local_files_only=True, # load files from local cache
    )
    pipe = pipe.to(accelerator.device)
    pipe.set_progress_bar_config(disable=True)

    if args.enable_xformers_memory_efficient_attention:
        pipe.enable_xformers_memory_efficient_attention()

    # load validation images
    image_logs = []
    for case in args.validation_data.cases:
        validation_prompts = case.prompt
        validation_image = Image.open(os.path.join(args.validation_data.data_root, case.image)).convert("RGB")
        validation_mask = Image.open(os.path.join(args.validation_data.data_root, case.mask))
        validation_mask = validation_mask.resize((validation_image.size[0], validation_image.size[1]), Image.NEAREST)
        validation_mask = validation_mask.convert("L")
        hole_value = (0, 0, 0)
        validation_image = Image.composite(
            Image.new("RGB", (validation_image.size[0], validation_image.size[1]), hole_value),
            validation_image,
            validation_mask.convert("L"),
        )
        image_grid = Image.new(
            "RGB",
            (validation_image.size[0] * (1 + len(validation_prompts)), validation_image.size[1]),
            (255, 255, 255),
        )
        image_grid.paste(validation_image, (0, 0))
        for i, p in enumerate(validation_prompts):
            with torch.autocast(accelerator.device.type):
                image = pipe(
                    promptA=p.promptA,
                    promptB=p.promptB,
                    negative_promptA=p.get("negative_promptA", None),
                    negative_promptB=p.get("negative_promptB", None),
                    tradeoff=p.tradeoff,
                    image=validation_image,
                    mask=validation_mask,
                    num_inference_steps=45,
                ).images[0]
            image_logs.append(image)
            image_grid.paste(image, (validation_image.size[0] * (i + 1), 0))
        save_path = os.path.join(
            args.output_dir,
            f"{case.name}_{str(step).zfill(3)}_{os.path.basename(case.image)}"
        )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        image_grid.save(save_path)

       # image_grid.save(os.path.join(args.output_dir, f"{case.name}_{str(step).zfill(3)}_{os.path.basename(case.image)}"))
    gc.collect()
    torch.cuda.empty_cache()

    # for tracker in accelerator.trackers:
    #     if tracker.name == "tensorboard":
    #         np_images = np.stack([np.asarray(img) for img in image_logs])
    #         tracker.writer.add_images("validation", np_images, step, dataformats="NHWC")
    #     elif tracker.name == "wandb":
    #         tracker.log(
    #             {
    #                 "validation": [
    #                     wandb.Image(image, caption=f"{p.task}")
    #                     for image, p in zip(image_logs, args.validation_data.cases[0].prompt)
    #                 ]
    #             }
    #         )
    #     else:
    #         logger.warning(f"image logging not implemented for {tracker.name}")

    del pipe
    torch.cuda.empty_cache()

    return image_logs

def parse_args():
    pass

def main():
    args = parse_args()
    if args.non_ema_revision is not None:
        deprecate(
            "non_ema_revision!=None",
            "0.15.0",
            message=(
                "Downloading 'non_ema' weights from revision branches of the Hub is deprecated. Please make sure to"
                " use `--variant=non_ema` instead."
            ),
        )

    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

     # If passed along, set the training seed now.
    if args.seed is not None:
        torch.manual_seed(args.seed)
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        # saving training configuration to output_dir
        to_save_config = OmegaConf.create(vars(args))
        OmegaConf.save(config=to_save_config, f=os.path.join(args.output_dir, "training_config.yaml"))

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

    # ==========================================
    # setting models: load scheduler, tokenizer and models.
    # ==========================================
    pipe = StableDiffusionInpaintIndomainPipeline.from_pretrained(
        args.base_model_path,
        torch_dtype=weight_dtype,
        local_files_only=True
    )
    pipe.tokenizer = TokenizerWrapper(
        from_pretrained=args.base_model_path,
        subfolder='tokenizer',
        torch_dtype=weight_dtype,
        local_files_only=True
    )

    # add pretrained learned task tokens into the tokenizer
    add_tokens(
        tokenizer=pipe.tokenizer,
        text_encoder=pipe.text_encoder,
        placeholder_tokens=["P_ctxt", "P_shape", "P_obj"],
        initialize_tokens=["a", "a", "a", "a"],
        num_vectors_per_token=10,
    )
    # load ppt1 checkpoint
    load_model(pipe.unet, os.path.join(args.ppt1_checkpoint, "unet/unet.safetensors"), strict=False)
    load_model(pipe.text_encoder, os.path.join(args.ppt1_checkpoint, "text_encoder/text_encoder.safetensors"),strict=False )

    # add new learned task tokens
    add_tokens(
        tokenizer=pipe.tokenizer,
        text_encoder=pipe.text_encoder,
        placeholder_tokens=["P_loc"],
        initialize_tokens=["P_ctxt"],
        num_vectors_per_token=10,
    )

    vae, unet, tokenizer, noise_scheduler = pipe.vae, pipe.unet, pipe.tokenizer, pipe.scheduler
    text_encoder= pipe.text_encoder.to(torch.float32)


    # Freeze all parameters except for the token embeddings of p_loc
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.text_model.requires_grad_(True)
    text_encoder.text_model.encoder.requires_grad_(False)
    text_encoder.text_model.final_layer_norm.requires_grad_(False)
    text_encoder.text_model.embeddings.position_embedding.requires_grad_(False)
    text_encoder.text_model.embeddings.token_embedding.trainable_embeddings.P_ctxt.requires_grad_(False)
    text_encoder.text_model.embeddings.token_embedding.trainable_embeddings.P_obj.requires_grad_(False)
    text_encoder.text_model.embeddings.token_embedding.trainable_embeddings.P_shape.requires_grad_(False)
    text_encoder.text_model.embeddings.token_embedding.wrapped.weight.requires_grad_(False)

    # Check what we set grad True
    for name, p in text_encoder.named_parameters():
        if p.requires_grad:
            logger.info(f'{name} {p.shape}')

    log_validation(tokenizer, text_encoder, unet, args, accelerator, weight_dtype, 0)

    return
    # # Don't need to create EMA for the unet because we just finetune text encoder
    # if args.use_ema:
    #     ema_unet = UNet2DConditionModel.from_pretrained(
    #         args.ppt1_checkpoint, subfolder="unet", torch_dtype=weight_dtype, local_files_only=True
    #     )
    #     ema_unet = EMAModel(ema_unet.parameters(), model_cls=UNet2DConditionModel, model_config=ema_unet.config)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # # Taken from [Sayak Paul's Diffusers PR #6511](https://github.com/huggingface/diffusers/pull/6511/files)
    # def unwrap_model(model):
    #     model = accelerator.unwrap_model(model)
    #     model = model._orig_mod if is_compiled_module(model) else model
    #     return model
    
    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                for model in models:
                    sub_dir = "text_encoder"
                    model.save_pretrained(os.path.join(output_dir, sub_dir))

                    # make sure to pop weight so that corresponding model is not saved again
                    weights.pop()

        # def load_model_hook(models, input_dir):
        #     while len(models) > 0:
        #         model = models.pop()

        #         if isinstance(model, type(unwrap_model(text_encoder))):
        #             # load transformers style into model
        #             loaded_model = text_encoder_cls.from_pretrained(input_dir, subfolder="text_encoder")
        #             model.config = load_model.config

        #         model.load_state_dict(load_model.state_dict())
        #         del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        # accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        text_encoder.gradient_checkpointing_enable()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )
        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    # get trainable parameters
    embedding_layer = text_encoder.get_input_embeddings()
    trainable_prompt = embedding_layer.trainable_embeddings['P_loc']

    optimizer = optimizer_cls(
        list(trainable_prompt),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Preparing datasets and dataloader for training
    train_dataset = FSCDataset(args.train_data.datasets.data_path, transforms=None, pipeline=pipe, task_prompt=args.task_prompt, resolution=args.train_data.resolution)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        shuffle=True
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader)/args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs*num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    unet, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, text_encoder, optimizer, train_dataloader, lr_scheduler
    )

    # Move vae to gpu and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))

        # tensorboard cannot handle list types for config
        pop_list = []
        for k, v in tracker_config.items():
            if not isinstance(v, (int, float, str, bool, torch.Tensor)):
                pop_list.append(k)
                logger.info(f"Removed {k} (type:{type(v)}) from tracker_config")
        for k in pop_list:
            tracker_config.pop(k)

        accelerator.init_trackers(args.tracker_project_name, tracker_config)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info(f"***** Running training for {args.tracker_project_name} *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {int(args.max_train_steps)}")
    global_step = 0
    first_epoch = 0

    # Only show the progress bar once on each machine.args.max_train_steps
    progress_bar = tqdm(
        range(0, int(args.max_train_steps)),
        initial=global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    text_encoder.train()

    for _ in range(first_epoch, args.num_train_epochs):
        train_loss = 0.0

        for batch in train_dataloader:

            with accelerator.accumulate(text_encoder):
                # Convert images to latent space
                latents = vae.encode(batch["pixel_values"].to(weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                if args.noise_offset:
                    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                    noise += args.noise_offset * torch.randn(
                        (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
                    )
                if args.input_perturbation:
                    new_noise = noise + args.input_perturbation * torch.randn_like(noise)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # in mask, 1 for masked region, 0 for known region
                mask_image = batch["pixel_values"] * (batch["mask"] < 0.5)
                # convert the hole value from 0 to -1 due to value range [-1, 1]
                mask_image = mask_image - batch["mask"]
                mask_image_latents = vae.encode(mask_image.to(weight_dtype)).latent_dist.sample()
                mask_image_latents = mask_image_latents * vae.config.scaling_factor

                mask = torch.nn.functional.interpolate(batch["mask"], size=(64, 64))

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                if args.input_perturbation:
                    noisy_latents = noise_scheduler.add_noise(latents, new_noise, timesteps)
                else:
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                model_input = torch.cat([noisy_latents, mask, mask_image_latents], dim=1)

                # Get the text embedding for conditioning unet, (bs, 77, 768)
                encoder_hidden_statesA = text_encoder(batch["input_idsA"], return_dict=False)[0]
                encoder_hidden_statesB = text_encoder(batch["input_idsB"], return_dict=False)[0]
                tradeoff = batch["tradeoff"].unsqueeze(-1)
                encoder_hidden_states = (
                    tradeoff[:, 0:1, :] * encoder_hidden_statesA + tradeoff[:, 1:, :] * encoder_hidden_statesB.detach()
                )
                encoder_hidden_states = encoder_hidden_states.to(accelerator.unwrap_model(unet).dtype)

                # Get the target for loss depending on the prediction type
                if args.prediction_type is not None:
                    # set prediction_type of scheduler if defined
                    noise_scheduler.register_to_config(prediction_type=args.prediction_type)

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                # Predict the noise residual and compute loss
                model_pred = unet(model_input, timesteps, encoder_hidden_states, return_dict=False)[0]

                if args.snr_gamma is None:
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                else:
                    # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                    # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                    # This is discussed in Section 4.2 of the same paper.
                    snr = compute_snr(timesteps)
                    mse_loss_weights = (
                        torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
                    )
                    # We first calculate the original loss. Then we mean over the non-batch dimensions and
                    # rebalance the sample-wise losses with their respective loss weights.
                    # Finally, we take the mean of the rebalanced loss.
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                    loss = loss.mean()

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps


                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_prompt, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Check if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        unwrapped_text_encoder = accelerator.unwrap_model(text_encoder)
                        # Chỉ trích xuất state_dict của phần trainable
                        trainable_state_dict = {
                            k: v for k, v in unwrapped_text_encoder.state_dict().items()
                            if "trainable_embeddings" in k
                        }
                        save_path = os.path.join(args.output_dir, f"task_prompt_step_{global_step}.pt")
                        accelerator.save(trainable_state_dict, save_path)
                        logger.info(f"Saved state to {save_path}")
                    
                    if hasattr(args, "validation_data") is not None and global_step % args.validation_steps == 0:
                        log_validation(
                            tokenizer,
                            text_encoder,
                            unet,
                            args,
                            accelerator,
                            weight_dtype,
                            global_step,
                        )

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            if hasattr(args, "validation_data"):
                logger.info("Running inference...")
                log_validation(tokenizer, text_encoder, unet, args, accelerator, weight_dtype, global_step)

        accelerator.end_training()

if __name__ == "__main__":
    main()




