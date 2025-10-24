#!/usr/bin/env python
"""
accelerate launch train_controlnet_facemask.py \
  --data_root              ./your_face_dataset \
  --output_dir             ./out_controlnet_face \
  --batch_size             4 \
  --max_steps              50000 \
  --validation_every       1000
```
"""
from __future__ import annotations
import argparse, os, math, re
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from diffusers import AutoencoderKL, ControlNetModel, DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import random

try:
    import bitsandbytes as bnb
except ImportError:
    bnb = None

# ---------------- Custom Dataset for Face Masks -----------------
class FaceMaskDataset(Dataset):
    def __init__(self, root: str | Path, required_mask_indices: list[int], split: str = 'train', val_split_ratio: float = 0.1, res=512):
        self.root = Path(root)
        self.res = res
        self.split = split
        self.required_mask_indices = sorted(required_mask_indices)
        self.num_control_channels = len(self.required_mask_indices)

        target_dir = self.root / "input"
        control_base_dir = self.root / "output" / "resnet34"

        # Find all potential target images
        all_target_files = list(target_dir.glob("*.png"))
        
        # Filter for valid pairs that have all required masks
        self.valid_pairs = []
        print("Scanning dataset for valid image-mask pairs...")
        for target_file in tqdm(all_target_files):
            stem = target_file.stem
            mask_dir = control_base_dir / stem
            if mask_dir.is_dir():
                has_all_masks = True
                for index in self.required_mask_indices:
                    if not (mask_dir / f"{index}.png").exists():
                        has_all_masks = False
                        break
                if has_all_masks:
                    self.valid_pairs.append((target_file, mask_dir))

        if not self.valid_pairs:
            raise FileNotFoundError(f"No valid image-mask pairs found in {self.root}. Please check paths and required indices.")

        # Create a reproducible train/validation split
        random.Random(42).shuffle(self.valid_pairs)
        split_idx = int(len(self.valid_pairs) * (1 - val_split_ratio))
        
        if self.split == 'train':
            self.image_pairs = self.valid_pairs[:split_idx]
            print(f"Found {len(self.image_pairs)} valid training image pairs.")
        elif self.split == 'val':
            self.image_pairs = self.valid_pairs[split_idx:]
            print(f"Found {len(self.image_pairs)} valid validation image pairs.")
        else:
            raise ValueError(f"Invalid split '{self.split}'. Must be 'train' or 'val'.")

        self.tf_to_tensor = transforms.ToTensor()
        self.tf_resize = transforms.Resize((res, res), interpolation=transforms.InterpolationMode.BICUBIC)
        self.tf_norm = transforms.Normalize([0.5], [0.5]) # For single channel masks
        self.tf_norm_rgb = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])


    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, i):
        target_path, mask_dir_path = self.image_pairs[i]

        try:
            # --- Load and process the target image (ground truth face) ---
            target_img = Image.open(target_path).convert("RGB")
            target_img_resized = self.tf_resize(target_img)
            target_tensor = self.tf_to_tensor(target_img_resized)
            target_tensor = self.tf_norm_rgb(target_tensor)

            # --- Load, process, and stack the control masks ---
            mask_tensors = []
            for index in self.required_mask_indices:
                mask_path = mask_dir_path / f"{index}.png"
                mask_img = Image.open(mask_path).convert("L") # Load as grayscale
                mask_img_resized = self.tf_resize(mask_img)
                mask_tensor = self.tf_to_tensor(mask_img_resized)
                # No normalization needed if masks are already 0-1
                mask_tensors.append(mask_tensor)
            
            # Stack masks into a single multi-channel tensor
            control_tensor = torch.cat(mask_tensors, dim=0)

            return {"rgb": target_tensor, "control_image": control_tensor}
        except Exception as e:
            print(f"Error loading item {i} ({target_path}): {e}")
            # Return a dummy item or skip
            return self.__getitem__((i + 1) % len(self))


# ------------- Helpers -------------
@torch.no_grad()
def encode_latents(vae, img_batch):
    return vae.encode(img_batch.to(dtype=vae.dtype)).latent_dist.sample() * vae.config.scaling_factor

def plot_loss(filename, title="Loss Plot", **series):
    plt.figure(figsize=(12, 8))
    for label, (steps, values) in series.items():
        if not values: continue
        marker = 'o' if 'Validation' in label else None
        linestyle = '--' if 'Validation' in label else '-'
        plt.plot(steps, values, label=label, marker=marker, linestyle=linestyle, markersize=4)
    plt.xlabel("Training Steps"); plt.ylabel("Loss"); plt.title(title)
    plt.legend(); plt.grid(True, alpha=0.4); plt.savefig(filename); plt.close()

# ------------- Main Training Function -------------
def train(cfg):
    accelerator = Accelerator(mixed_precision=cfg.mixed_precision, log_with="tensorboard", project_config=ProjectConfiguration(project_dir=cfg.output_dir, logging_dir=Path(cfg.output_dir, "logs")))
    device = accelerator.device
    
    # Define the required mask indices based on your project needs
    # User specified: 2, 3, 4, 5, 10, 12, 13, 17
    required_mask_indices = [2, 3, 4, 5, 10, 12, 13, 17]
    num_control_channels = len(required_mask_indices)
    print(f"Training with {num_control_channels} control channels: {required_mask_indices}")

    unet = UNet2DConditionModel.from_pretrained(cfg.base_model, subfolder="unet")
    vae = AutoencoderKL.from_pretrained(cfg.base_model, subfolder="vae")

    if cfg.load_controlnet_from:
        print(f"Loading existing ControlNet from: {cfg.load_controlnet_from}")
        controlnet = ControlNetModel.from_pretrained(cfg.load_controlnet_from)
    else:
        print("Initializing new ControlNet from UNet...")
        controlnet = ControlNetModel.from_unet(unet, conditioning_channels=num_control_channels)

    vae.requires_grad_(False); unet.requires_grad_(False)
    if cfg.use_gradient_checkpointing: controlnet.enable_gradient_checkpointing()

    optimizer_cls = bnb.optim.AdamW8bit if cfg.use_8bit_adam and bnb else torch.optim.AdamW
    optimizer = optimizer_cls(controlnet.parameters(), lr=cfg.lr, weight_decay=cfg.adam_weight_decay)
    
    noise_scheduler = DDPMScheduler.from_pretrained(cfg.base_model, subfolder="scheduler")
    lr_scheduler = get_scheduler(cfg.lr_scheduler, optimizer=optimizer, num_warmup_steps=cfg.warmup_steps, num_training_steps=cfg.max_steps)

    train_dataset = FaceMaskDataset(cfg.data_root, required_mask_indices, split='train')
    val_dataset = FaceMaskDataset(cfg.data_root, required_mask_indices, split='val')
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    controlnet, optimizer, train_dataloader, val_dataloader, lr_scheduler, unet, vae = accelerator.prepare(
        controlnet, optimizer, train_dataloader, val_dataloader, lr_scheduler, unet, vae)
    ema_controlnet = EMAModel(accelerator.unwrap_model(controlnet).parameters(), decay=cfg.ema_decay) if cfg.use_ema else None

    global_step, ema_loss = 0, None
    loss_history, step_history = [], []
    val_loss_history, val_step_history = [], []

    if cfg.resume_from_checkpoint:
        print(f"Resuming from checkpoint: {cfg.resume_from_checkpoint}")
        accelerator.load_state(cfg.resume_from_checkpoint)
        global_step = int(re.findall(r"\d+", Path(cfg.resume_from_checkpoint).name)[-1])
        
    progress_bar = tqdm(range(global_step, cfg.max_steps), initial=global_step, total=cfg.max_steps, disable=not accelerator.is_local_main_process)
    
    for epoch in range(math.ceil(cfg.max_steps / len(train_dataloader))):
        if global_step >= cfg.max_steps: break
        for batch in train_dataloader:
            controlnet.train()
            with accelerator.accumulate(controlnet):
                target_rgb, control_image = batch["rgb"], batch["control_image"]

                latents = encode_latents(vae, target_rgb)
                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                # Using empty prompts as this is an unconditional ControlNet
                prompt_embeds = torch.zeros((latents.shape[0], 77, unet.config.cross_attention_dim), device=device, dtype=unet.dtype)

                down_samples, mid_sample = controlnet(noisy_latents, timesteps, prompt_embeds, control_image, return_dict=False)
                model_pred = unet(noisy_latents, timesteps, prompt_embeds, down_block_additional_residuals=down_samples, mid_block_additional_residual=mid_sample).sample

                loss = F.mse_loss(model_pred.float(), noise.float())

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(controlnet.parameters(), 1.0)
                optimizer.step(); lr_scheduler.step(); optimizer.zero_grad(set_to_none=True)
                if ema_controlnet: ema_controlnet.step(controlnet.parameters())
                
            global_step += 1; progress_bar.update(1)
            
            if accelerator.is_main_process:
                ema_loss = loss.item() if ema_loss is None else 0.9 * ema_loss + 0.1 * loss.item()
                loss_history.append(ema_loss); step_history.append(global_step)
                progress_bar.set_postfix({"loss": f"{ema_loss:.4f}"})
                accelerator.log({"train_loss": ema_loss}, step=global_step)
                
                if global_step % cfg.validation_every == 0:
                    controlnet.eval()
                    val_losses = []
                    for val_batch in val_dataloader:
                        val_target_rgb, val_control_image = val_batch["rgb"], val_batch["control_image"]
                        val_latents = encode_latents(vae, val_target_rgb)
                        val_noise = torch.randn_like(val_latents)
                        val_timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (val_latents.shape[0],), device=device).long()
                        val_noisy_latents = noise_scheduler.add_noise(val_latents, val_noise, val_timesteps)
                        val_prompt_embeds = torch.zeros((val_latents.shape[0], 77, unet.config.cross_attention_dim), device=device, dtype=unet.dtype)
                        down, mid = controlnet(val_noisy_latents, val_timesteps, val_prompt_embeds, val_control_image, return_dict=False)
                        pred = unet(val_noisy_latents, val_timesteps, val_prompt_embeds, down_block_additional_residuals=down, mid_block_additional_residual=mid).sample
                        val_loss = F.mse_loss(pred.float(), val_noise.float())
                        val_losses.append(val_loss.item())
                    
                    avg_val_loss = np.mean(val_losses)
                    val_loss_history.append(avg_val_loss)
                    val_step_history.append(global_step)
                    accelerator.log({"val_loss": avg_val_loss}, step=global_step)
                    progress_bar.set_postfix({"loss": f"{ema_loss:.4f}", "val_loss": f"{avg_val_loss:.4f}"})
                    plot_loss(Path(cfg.output_dir) / "loss_plot.png", "Loss", Training=(step_history, loss_history), Validation=(val_step_history, val_loss_history))

                if global_step > 0 and global_step % cfg.save_checkpoint_every == 0:
                    save_path = Path(cfg.output_dir) / f"checkpoint_step_{global_step}"
                    accelerator.save_state(save_path)
                    print(f"\nSaved checkpoint to {save_path}")

            if global_step >= cfg.max_steps: break
        if global_step >= cfg.max_steps: break

    if accelerator.is_main_process:
        save_path = Path(cfg.output_dir) / "final_model"
        model_to_save = accelerator.unwrap_model(controlnet)
        if ema_controlnet: ema_controlnet.copy_to(model_to_save.parameters())
        model_to_save.save_pretrained(save_path)
        print(f"Saved final model to {save_path}")
    accelerator.end_training()

# ------------- CLI Argument Parser -------------
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train ControlNet on face masks.")
    p.add_argument("--data_root", type=str, required=True, help="Root directory of the dataset.")
    p.add_argument("--output_dir", type=str, required=True, help="Directory to save checkpoints and logs.")
    p.add_argument("--base_model", type=str, default="runwayml/stable-diffusion-v1-5")
    p.add_argument("--resume_from_checkpoint", type=str, default=None)
    p.add_argument("--load_controlnet_from", type=str, default=None, help="Path to a pre-trained ControlNet to continue training.")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_steps", type=int, default=50000)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--adam_weight_decay", type=float, default=1e-2)
    p.add_argument("--lr_scheduler", type=str, default="constant", help="one of [linear, cosine, constant, constant_with_warmup]")
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--use_ema", action="store_true", help="Use Exponential Moving Average of model weights.")
    p.add_argument("--ema_decay", type=float, default=0.9999)
    p.add_argument("--save_checkpoint_every", type=int, default=2500)
    p.add_argument("--validation_every", type=int, default=1000, help="Run validation every N steps.")
    p.add_argument("--use_8bit_adam", action="store_true", help="Use 8-bit Adam optimizer from bitsandbytes.")
    p.add_argument("--use_gradient_checkpointing", action="store_true")
    p.add_argument("--mixed_precision", choices=["no", "fp16", "bf16"], default="fp16")
    
    cfg = p.parse_args()
    os.makedirs(cfg.output_dir, exist_ok=True)
    train(cfg)
