import argparse
import gc
import glob
import logging
import math
import os
import random
import shutil
from pathlib import Path
from dataset import TextPromptDataset
import accelerate
import diffusers
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.lr_scheduler import CosineAnnealingLR
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from packaging import version
from peft import LoraConfig, PeftModel
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig
from diffusers import DDIMScheduler

check_min_version("0.24.0.dev0")

logger = get_logger(__name__)


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder="text_encoder", revision=revision
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--validation_prompts",
        type=str,
        default=None,
        nargs="+",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--proportion_empty_prompts",
        type=float,
        default=0,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--learning_rate_lora",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )

    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true"
    )
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=0.0)
    parser.add_argument("--adam_epsilon", type=float, default=1e-08)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
    )
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true"
    )
    parser.add_argument(
        "--set_grads_to_none",
        action="store_true",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.train_data_dir is None:
        raise ValueError("Need a training folder.")

    return args


def encode_prompt(prompts, text_encoder, tokenizer, is_train=True):
    captions = []
    for caption in prompts:
        if isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            captions.append(random.choice(caption) if is_train else caption[0])

    with torch.no_grad():
        text_inputs = tokenizer(
            captions,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        prompt_embeds = text_encoder(
            text_input_ids.to(text_encoder.device),
        )[0]

    return {"prompt_embeds": prompt_embeds.cpu()}


@torch.no_grad()
def inference(vae, unet, noise_scheduler, encoded_embeds, generator, device, weight_dtype):
    input_shape = (1, 4, args.resolution // 8, args.resolution // 8)
    input_noise = torch.randn(input_shape, generator=generator, device=device, dtype=weight_dtype)

    prompt_embed = encoded_embeds["prompt_embeds"]
    prompt_embed = prompt_embed.to(device, weight_dtype)

    pred_original_sample = predict_original(unet, noise_scheduler, input_noise, prompt_embed)
    pred_original_sample = pred_original_sample / vae.config.scaling_factor

    image = vae.decode(pred_original_sample.to(dtype=vae.dtype)).sample.float()
    image = (image[0].detach().cpu() * 127.5 + 128).clamp(0, 255).to(torch.uint8)

    return image


def predict_original(unet, noise_scheduler, input_noise, prompt_embeds):
    max_timesteps = torch.ones((input_noise.shape[0],), dtype=torch.int64, device=input_noise.device)
    max_timesteps = max_timesteps * (noise_scheduler.config.num_train_timesteps - 1)

    alpha_T, sigma_T = 0.0047**0.5, (1 - 0.0047) ** 0.5
    model_pred = unet(input_noise, max_timesteps, prompt_embeds).sample

    latents = (input_noise - sigma_T * model_pred) / alpha_T
    return latents


class PromptDataset(Dataset):
    def __init__(self, train_data_dir):
        self.train_data_paths = list(glob.glob(train_data_dir + "/*.npy"))

    def __len__(self):
        return len(self.train_data_paths)

    def __getitem__(self, index):
        data = {"prompt_embeds": torch.from_numpy(np.load(self.train_data_paths[index], allow_pickle=True))}
        return data

    def shuffle(self, *args, **kwargs):
        random.shuffle(self.train_data_paths)
        return self

    def select(self, selected_range):
        self.train_data_paths = [self.train_data_paths[idx] for idx in selected_range]
        return self


def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
        project_config=accelerator_project_config,
    )

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

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision, use_fast=False
    )

    text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)

    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    text_encoder = text_encoder_cls.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )

    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)

    teacher_lora = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision,
        torch_dtype=torch.float16
    )
    teacher = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision,
        torch_dtype=torch.float16
    )

    
    def hook_fn(module, input, output):
        if isinstance(module, nn.Linear):
            if module.weight.dim() == 2:
                W = module.weight
                d = W.shape[1] 

                if d % 2 != 0:
                    raise ValueError("Input feature dimension must be even for LoRaD.")

                W_odd = W[:, 0::2]
                W_even = W[:, 1::2]

                theta = module.theta_a @ module.theta_b
                theta = ((theta + torch.pi) % (2 * torch.pi)) - torch.pi
               
                cos_theta = torch.cos(theta)  
                sin_theta = torch.sin(theta)

              
                W_rotated_odd = W_odd * cos_theta - W_even * sin_theta
                W_rotated_even = W_odd * sin_theta + W_even * cos_theta
                
               
                W_rotated = torch.empty_like(W)
                W_rotated[:, 0::2] = W_rotated_odd  
                W_rotated[:, 1::2] = W_rotated_even 
                W_rotated = W_rotated
                output = F.linear(input[0], W_rotated)

                if module.bias is not None:
                    output += module.bias.unsqueeze(0).expand_as(output)
                return output


    def register_hooks(model, rank=16):
        skip_layers = ["conv_shortcut"]
        for name, module in model.named_modules():
            if "up_blocks" in name or "down_blocks" in name or "mid_block" in name:
                if isinstance(module, nn.Linear) and not any(skip_layer in name for skip_layer in skip_layers):
                    in_features = module.weight.shape[1]
                    out_features = module.weight.shape[0]
                    if in_features % 2 != 0:
                        raise ValueError(f"Input feature dimension {in_features} must be even for LoRaD.")
                    
                    module.theta_a = nn.Parameter(
                        torch.zeros(out_features, rank, dtype=torch.float32, device=module.weight.device),
                        requires_grad=True
                    )
                    module.theta_b = nn.Parameter(
                        torch.zeros(rank, in_features // 2, dtype=torch.float32, device=module.weight.device),
                        requires_grad=True
                    )
                    nn.init.xavier_uniform_(module.theta_b)
                    module.register_forward_hook(hook_fn)

    def register_hooks_teacher(model, rank=16):
        skip_layers = ["proj_in", "proj_out", "time_emb_proj"]
        for name, module in model.named_modules():
            if "up_blocks" in name or "down_blocks" in name or "mid_block" in name:
                if isinstance(module, nn.Linear) and not any(skip_layer in name for skip_layer in skip_layers):
                    
                    in_features = module.weight.shape[1]
                    out_features = module.weight.shape[0]
                    if in_features % 2 != 0:
                        raise ValueError(f"Input feature dimension {in_features} must be even for LoRaD.")
                   
                    module.theta_a = nn.Parameter(
                        torch.zeros(out_features, rank, dtype=torch.float32, device=module.weight.device),
                        requires_grad=True
                    )
                    module.theta_b = nn.Parameter(
                        torch.zeros(rank, in_features // 2, dtype=torch.float32, device=module.weight.device),
                        requires_grad=True
                    )
                    nn.init.xavier_uniform_(module.theta_b)
                    module.register_forward_hook(hook_fn)

    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision,
        torch_dtype=torch.float16
    )

    register_hooks(unet, rank=256)
    register_hooks_teacher(teacher_lora, rank=32)
    unet.config.model_type = "unet_lora"
    teacher_lora.config.model_type = "teacher_lora"  

    for name, param in unet.named_parameters():
        if "theta" not in name:
            param.requires_grad = False

    for name, param in teacher_lora.named_parameters():
        if "theta" not in name:
            param.requires_grad = False

    def count_trainable_params(model):
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total_params / 1e6  

    trainable_params = count_trainable_params(unet)
    print(f"The number of trainable parameters: {trainable_params:.2f}M")
    trainable_params = count_trainable_params(teacher_lora)
    print(f"The number of trainable parameters: {trainable_params:.2f}M")

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(accelerator.device, dtype=torch.float32)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    if args.use_ema:
        ema_theta_dict = {}
        for name, module in unet.named_modules():
            if hasattr(module, "theta_a"):
                ema_theta_dict[f"{name}.theta_a"] = module.theta_a.clone().to(accelerator.device)
                ema_theta_dict[f"{name}.theta_b"] = module.theta_b.clone().to(accelerator.device)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs."
                )
            unet.enable_xformers_memory_efficient_attention()
            teacher_lora.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                for i, model in enumerate(models):
                    model_type = getattr(model.config, "model_type", None)
                    if model_type == "unet_lorad":
                        subfolder = "kdro"
                        path = os.path.join(output_dir, subfolder)
                        os.makedirs(path, exist_ok=True)
                        hook_params = {name: param for name, param in model.state_dict().items() if "theta" in name}
                        torch.save(hook_params, os.path.join(path, "kduv.pth"))
                        if args.use_ema:
                            torch.save(ema_theta_dict, os.path.join(path, "ema_kduv.pth"))
                        
                        
                    elif model_type == "teacher_lorad":
                        subfolder = "teacher_lorad"
                        
                        path = os.path.join(output_dir, subfolder)
                        os.makedirs(path, exist_ok=True)
                       
                        hook_params = {name: param for name, param in model.state_dict().items() if "theta" in name}
                        torch.save(hook_params, os.path.join(path, "kduv.pth"))

                    weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), UNet2DConditionModel)
                ema_unet.load_state_dict(load_model.state_dict())
                ema_unet.to(accelerator.device)
                del load_model

            for model in models:
                model_type = getattr(model.config, "model_type", None)
                if model_type == "unet_lorad":
                    base_model = UNet2DConditionModel.from_pretrained(
                        args.pretrained_model_name_or_path, subfolder="unet"
                    )
                    load_path = os.path.join(input_dir, "teacher_lorad")
                    load_path1 = os.path.join(load_path, "kduv.pth")
                    load_model = base_model.load_state_dict(torch.load(load_path1), strict=False)
                elif model_type == "teacher_lorad":
                    base_model = UNet2DConditionModel.from_pretrained(
                        args.pretrained_model_name_or_path, subfolder="unet"
                    )
                    load_path = os.path.join(input_dir, "kdro")
                    load_path1 = os.path.join(load_path, "kduv.pth")
                    load_model = base_model.load_state_dict(torch.load(load_path1), strict=False)

                model.register_to_config(**load_model.config)
                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        teacher_lora.enable_gradient_checkpointing()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

        args.learning_rate_lora = (
            args.learning_rate_lora
            * args.gradient_accumulation_steps
            * args.train_batch_size
            * accelerator.num_processes
        )

    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
        optimizer_lora_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW
        optimizer_lora_class = torch.optim.AdamW

    optimizer = optimizer_class(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    optimizer_lora = optimizer_lora_class(
        teacher_lora.parameters(),
        lr=args.learning_rate_lora,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    with accelerator.main_process_first():
        train_dataset = TextPromptDataset(args.train_data_dir)
        train_dataset.text_encoder, train_dataset.tokenizer = text_encoder, tokenizer
        if args.max_train_samples is not None:
            train_dataset = train_dataset.shuffle(seed=args.seed).select(range(args.max_train_samples))

    if type(args.validation_prompts) is not list:
        args.validation_prompts = [args.validation_prompts]

    null_dict = encode_prompt([""], text_encoder, tokenizer)
    validation_dicts = [encode_prompt([prompt], text_encoder, tokenizer) for prompt in args.validation_prompts]

    del text_encoder, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    def collate_fn(examples):
        prompt_embeds = torch.stack([example["prompt_embeds"] for example in examples])

        return {
            "prompt_embeds": prompt_embeds,
        }

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )
    


    lr_scheduler_lora = get_scheduler(
        "constant",
        optimizer=optimizer_lora,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    unet, teacher_lora, teacher, optimizer, optimizer_lora, train_dataloader, lr_scheduler, lr_scheduler_lora = (
        accelerator.prepare(
            unet, teacher_lora, teacher,optimizer, optimizer_lora, train_dataloader, lr_scheduler, lr_scheduler_lora
        )
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        tracker_config.pop("validation_prompts")
        accelerator.init_trackers("DKD", config=tracker_config)


    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0


    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    alphas_cumprod = noise_scheduler.alphas_cumprod
    alphas_cumprod = alphas_cumprod.to(accelerator.device, dtype=weight_dtype)

    for epoch in range(first_epoch, args.num_train_epochs):
        train_loss_vsd = train_loss_lora = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet, teacher_lora):
                bsz = batch["prompt_embeds"].shape[0]

                input_shape = (bsz, 4, args.resolution // 8, args.resolution // 8)
                input_noise = torch.randn(*input_shape, dtype=weight_dtype, device=accelerator.device)

                prompt_embeds = batch["prompt_embeds"].to(accelerator.device, dtype=weight_dtype)
                prompt_null_embeds = (
                    null_dict["prompt_embeds"].repeat(bsz, 1, 1).to(accelerator.device, dtype=weight_dtype)
                )

                pred_original_samples = predict_original(unet, noise_scheduler, input_noise, prompt_embeds).to(
                    dtype=weight_dtype
                )


                noise = torch.randn_like(pred_original_samples)

                timesteps_range = torch.tensor([0.02, 0.981]) * noise_scheduler.config.num_train_timesteps
                timesteps = torch.randint(*timesteps_range.long(), (bsz,), device=accelerator.device).long()

                noisy_samples = noise_scheduler.add_noise(pred_original_samples, noise, timesteps)

                with torch.no_grad():

                    accelerator.unwrap_model(teacher)
                    teacher_pred_cond = teacher(noisy_samples, timesteps, prompt_embeds).sample
                    teacher_pred_uncond = teacher(noisy_samples, timesteps, prompt_null_embeds).sample

                    accelerator.unwrap_model(teacher_lora)
                    lora_pred_cond = teacher_lora(noisy_samples, timesteps, prompt_embeds).sample
                    lora_pred_uncond = teacher_lora(noisy_samples, timesteps, prompt_null_embeds).sample

                    teacher_pred = teacher_pred_uncond + args.guidance_scale * (
                        teacher_pred_cond - teacher_pred_uncond
                    )
                    lora_pred = lora_pred_uncond + args.guidance_scale * (lora_pred_cond - lora_pred_uncond)

                sigma_t = ((1 - alphas_cumprod[timesteps]) ** 0.5).view(-1, 1, 1, 1)
                score_gradient = torch.nan_to_num(sigma_t**2 * (teacher_pred - lora_pred))

                target = (pred_original_samples - score_gradient).detach()
                loss_vsd = 0.5 * F.mse_loss(pred_original_samples.float(), target.float(), reduction="mean")

                avg_loss_vsd = accelerator.gather(loss_vsd.repeat(args.train_batch_size)).mean()
                train_loss_vsd += avg_loss_vsd.item() / args.gradient_accumulation_steps

                accelerator.backward(loss_vsd)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

                noise = torch.randn_like(pred_original_samples.detach())

                timesteps_range = torch.tensor([0, 1]) * noise_scheduler.config.num_train_timesteps
                timesteps = torch.randint(*timesteps_range.long(), (bsz,), device=accelerator.device).long()

                noisy_samples = noise_scheduler.add_noise(pred_original_samples.detach(), noise, timesteps)

                encoder_hidden_states = prompt_null_embeds if random.random() < 0.1 else prompt_embeds
                lora_pred = teacher_lora(noisy_samples, timesteps, encoder_hidden_states).sample


                alpha_t = (alphas_cumprod[timesteps] ** 0.5).view(-1, 1, 1, 1)
                lora_pred = alpha_t * lora_pred
                target = alpha_t * noise

                loss_lora = 1.0 * F.mse_loss(lora_pred.float(), target.float(), reduction="mean")

                avg_loss_lora = accelerator.gather(loss_lora.repeat(args.train_batch_size)).mean()
                train_loss_lora += avg_loss_lora.item() / args.gradient_accumulation_steps

                accelerator.backward(loss_lora)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(teacher_lora.parameters(), args.max_grad_norm)
                optimizer_lora.step()
                lr_scheduler_lora.step()
                optimizer_lora.zero_grad(set_to_none=args.set_grads_to_none)

            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_decay = 0.999
                    unet_ = accelerator.unwrap_model(unet)
                    for name, module in unet_.named_modules():
                        if hasattr(module, "theta_a"):
                            ema_theta_dict[f"{name}.theta_a"].mul_(ema_decay).add_(module.theta_a * (1 - ema_decay))
                            ema_theta_dict[f"{name}.theta_b"].mul_(ema_decay).add_(module.theta_b * (1 - ema_decay))

                progress_bar.update(1)
                global_step += 1
                accelerator.log(
                    {"train_loss_vsd": train_loss_vsd, "train_loss_lora": train_loss_lora}, step=global_step
                )
                train_loss_vsd = train_loss_lora = 0.0

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0 or global_step == args.max_train_steps  \
                            or global_step in [10, 200, 300, 400, 500, 800, 1000, 2000, 3000, 4000, 5000,6000, 8000,]:
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                    if global_step % args.validation_steps == 0 or global_step == args.max_train_steps:
                        if args.validation_prompts is not None and args.num_validation_images > 0:

                            logger.info(
                                "Running validation... \nGenerating {} images with prompts:\n  {}".format(
                                    args.num_validation_images, "\n  ".join(args.validation_prompts)
                                )
                            )

                            generator = (
                                torch.Generator(device=accelerator.device).manual_seed(args.seed)
                                if args.seed
                                else None
                            )

                            with torch.cuda.amp.autocast():
                                images = {}
                                for prompt, validation_dict in zip(args.validation_prompts, validation_dicts):
                                    images[prompt] = [
                                        inference(
                                            vae,
                                            unet,
                                            noise_scheduler,
                                            validation_dict,
                                            generator=generator,
                                            device=accelerator.device,
                                            weight_dtype=weight_dtype,
                                        )
                                        for _ in range(args.num_validation_images)
                                    ]


                            from PIL import Image
                            for tracker in accelerator.trackers:
                                for prompt in args.validation_prompts:
                                    for i, image in enumerate(images[prompt]):
                                        tracker.writer.add_images(
                                            f"{prompt}/{i}", np.asarray(image), global_step, dataformats="CHW"
                                        )

                                        output = f'{args.output_dir}/samples/{prompt}'
                                        if not os.path.exists(output):
                                            os.makedirs(output)

                                        from PIL import Image
                                        image_array = np.asarray(image).astype(np.uint8)
                                        image = Image.fromarray(np.transpose(image_array, (1, 2, 0)))

                                        image.save(f"{output}/{global_step}_{i}.png")

            logs = {
                "step_loss_vsd": loss_vsd.detach().item(),
                "step_loss_lora": loss_lora.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "lr_lora": lr_scheduler_lora.get_last_lr()[0],
            }
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)

        pipeline = DiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            unet=unet,
            revision=args.revision,
        )

        pipeline.scheduler = pipeline.scheduler.from_config(pipeline.scheduler.config)
        pipeline.save_pretrained(args.output_dir + "/final/")

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
