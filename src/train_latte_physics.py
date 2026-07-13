"""
Training script with PredictorP - Joint training of predictor + temporal cross-attention.

Trains:
1. PredictorP: Predicts VJEPA from (noisy_latents, text, timestep)
2. Temporal cross-attention: Uses predicted VJEPA for physics conditioning

Freezes:
- Spatial blocks
- Temporal self-attention
- All other Latte components
"""

import json
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
from accelerate import Accelerator
from diffusers import DDPMScheduler
import gc

from latte_physics import LatteTransformer3DModelWithPhysics

gc.collect()
torch.cuda.empty_cache()


def teacher_force_prob_at_step(step: int, total_steps: int, warmup_frac: float, anneal_frac: float) -> float:
    """p=1 for the first `warmup_frac` of training, linear 1->0 over the next
    `anneal_frac`, p=0 for the remainder. Anneals training away from GT VJEPA
    conditioning and onto PredictorP's own output."""
    if total_steps <= 0:
        return 0.0
    warmup_end = warmup_frac * total_steps
    anneal_end = (warmup_frac + anneal_frac) * total_steps
    if step < warmup_end:
        return 1.0
    if step >= anneal_end:
        return 0.0
    return 1.0 - (step - warmup_end) / max(anneal_end - warmup_end, 1e-9)

class PhysicsVideoDataset(Dataset):
    """Dataset with VJEPA, VAE latents, and text embeddings."""
    
    def __init__(self, index_json_path):
        with open(index_json_path, 'r') as f:
            self.data = json.load(f)
        self.video_ids = [data_dict["video_id"] for data_dict in self.data]
        
    def __len__(self):
        return len(self.video_ids)
    
    def __getitem__(self, idx):
        video_id = self.video_ids[idx]
        entry = self.data[idx]
        
        # VJEPA tokens [1, 2048, 1408]
        vjepa_data = np.load(entry['vjepa'])
        if 'arr_0' in vjepa_data:
            vfm_tokens = vjepa_data['arr_0']
        elif 'frames' in vjepa_data:
            vfm_tokens = vjepa_data['latents']
        else:
            vfm_tokens = vjepa_data[vjepa_data.files[0]]
        vjepa_tokens = torch.from_numpy(vfm_tokens).to(torch.bfloat16)
        vjepa_tokens = vjepa_tokens.squeeze(0).contiguous()
        
        # VAE latents [1, 16, 4, 32, 32]
        vae_data = np.load(entry['vae'])
        if 'arr_0' in vae_data:
            z0 = vae_data['arr_0']
        elif 'frames' in vae_data:
            z0 = vae_data['latents']
        else:
            z0 = vae_data[vae_data.files[0]]
        latents = torch.from_numpy(z0).to(torch.bfloat16)  # [1, 16, 4, 32, 32]
        latents = latents.permute(0, 2, 1, 3, 4).squeeze(0).contiguous() # [B, 4, 16, 32, 32]
        
        # Text embeddings [1, seq_len, 4096]
        text_embeddings = torch.from_numpy(np.load(entry['text'])).to(torch.bfloat16)
        text_embeddings = text_embeddings.squeeze(0).contiguous()
        
        return {
            'video_id': video_id,
            'latents': latents,  # [1, 16, 4, 32, 32]
            'vjepa_tokens': vjepa_tokens,  # [1, 2048, 1408]
            'text_embeddings': text_embeddings,  # [1, 226, 4096]
        }


def train_with_predictor(
    index_json_path: str,
    output_dir: str = "./latte_predictor_checkpoints",
    num_epochs: int = 3,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 1,
    learning_rate: float = 1e-5,
    vjepa_loss_weight: float = 1.0,  # Weight for VJEPA prediction loss
    mixed_precision: str = "bf16",
    num_train_samples: int = None,
    resume_from_checkpoint: str = None,
    tf_warmup_frac: float = 0.25,
    tf_anneal_frac: float = 0.5,
):
    """
    Joint training of PredictorP + temporal cross-attention layers.
    """

    accelerator = Accelerator(mixed_precision=mixed_precision,
                              gradient_accumulation_steps=gradient_accumulation_steps)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    print(f"Loading dataset from {index_json_path}...")
    full_dataset = PhysicsVideoDataset(index_json_path)
    
    if num_train_samples is not None:
        indices = list(range(min(num_train_samples, len(full_dataset))))
        dataset = torch.utils.data.Subset(full_dataset, indices)
        print(f"Using {len(dataset)} samples")
    else:
        dataset = full_dataset
        print(f"Using all {len(dataset)} samples")
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    # Initialize model with PredictorP
    print("\nInitializing Latte + PredictorP...")
    model = LatteTransformer3DModelWithPhysics(
        num_attention_heads=16,
        attention_head_dim=72,
        in_channels=4,
        out_channels=4,
        num_layers=28,
        dropout=0.0,
        cross_attention_dim=1152,
        attention_bias=False,
        sample_size=64,  # latent 64x64 -> pixel 512x512 (matches Latte-1 pretraining)
        patch_size=2,
        activation_fn="gelu-approximate",
        num_embeds_ada_norm=1000,
        norm_type="ada_norm_single",
        norm_elementwise_affine=False,
        norm_eps=1e-6,
        caption_channels=4096,
        video_length=16,
        # PredictorP config
        predictor_hidden_dim=512,
        vjepa_dim=1408,
        vjepa_seq_len=2048,  # 16 frames: 256 spatial * 8 temporal patches
        use_predictor=True,
    )
    
    # Try to load pretrained Latte base
    try:
        print("Loading pretrained Latte-1...")
        from diffusers import LatteTransformer3DModel
        pretrained = LatteTransformer3DModel.from_pretrained(
            "maxin-cn/Latte-1",
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
        )
        
        model_state = model.state_dict()
        pretrained_state = pretrained.state_dict()
        
        for key in pretrained_state:
            if key in model_state and model_state[key].shape == pretrained_state[key].shape:
                model_state[key] = pretrained_state[key]
        
        model.load_state_dict(model_state, strict=False)
        print("✓ Loaded pretrained Latte base")

        del pretrained
        del model_state
        del pretrained_state
        
        gc.collect()
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"Could not load pretrained: {e}")
    
    # ========== FREEZE BASE MODEL, TRAIN ONLY PHYSICS COMPONENTS ==========
    print("\n" + "="*60)
    print("FREEZING STRATEGY:")
    print("="*60)
    
    # Freeze everything first
    for param in model.parameters():
        param.requires_grad = False
    
    # 1. Unfreeze PredictorP (entire network)
    print("✓ Training: PredictorP (full network)")
    if model.predictor is not None:
        for param in model.predictor.parameters():
            param.requires_grad = True
    
    # 2. Unfreeze VJEPA projection
    print("✓ Training: VJEPA projection layer")
    if model.vjepa_projection is not None:
        for param in model.vjepa_projection.parameters():
            param.requires_grad = True
    
    # 3. Unfreeze temporal cross-attention (attn2 + norm2)
    print("✓ Training: Temporal cross-attention layers")
    num_blocks = len(model.temporal_transformer_blocks)
    blocks_to_train = 8

    for i, temp_block in enumerate(model.temporal_transformer_blocks):
        if i >= (num_blocks - blocks_to_train):
            if hasattr(temp_block, 'attn2'):
                for param in temp_block.attn2.parameters():
                    param.requires_grad = True
            if hasattr(temp_block, 'norm2'):
                for param in temp_block.norm2.parameters():
                    param.requires_grad = True
        else:
            if hasattr(temp_block, 'attn2'):
                for param in temp_block.attn2.parameters():
                    param.requires_grad = False
            if hasattr(temp_block, 'norm2'):
                for param in temp_block.norm2.parameters():
                    param.requires_grad = False
    
    # Print trainable parameter breakdown
    predictor_params = sum(p.numel() for p in model.predictor.parameters() if p.requires_grad) if model.predictor else 0
    vjepa_proj_params = sum(p.numel() for p in model.vjepa_projection.parameters() if p.requires_grad) if model.vjepa_projection else 0
    temporal_attn_params = sum(
        p.numel() for temp_block in model.temporal_transformer_blocks 
        for p in temp_block.parameters() if p.requires_grad
    )
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    print("\n" + "="*60)
    print("TRAINABLE PARAMETERS:")
    print("="*60)
    print(f"PredictorP:              {predictor_params:>12,} params")
    print(f"VJEPA projection:        {vjepa_proj_params:>12,} params")
    print(f"Temporal cross-attn:     {temporal_attn_params:>12,} params")
    print("-"*60)
    print(f"TOTAL TRAINABLE:         {total_trainable:>12,} params")
    print(f"TOTAL MODEL:             {total_params:>12,} params")
    print(f"Percentage trainable:    {100*total_trainable/total_params:>11.2f}%")
    print("="*60 + "\n")
    
    # Noise scheduler
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_schedule="scaled_linear",
        beta_start=0.0001,
        beta_end=0.02,
        clip_sample=False,
    )
    
    # Optimizer (only trainable params)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0.01,
    )
    
    start_epoch = 0
    global_step_offset = 0
    checkpoint_to_load = None

    # Resume from checkpoint if provided
    if resume_from_checkpoint:
        checkpoint_path = Path(resume_from_checkpoint)
        if checkpoint_path.is_dir():
            checkpoints = sorted(checkpoint_path.glob("checkpoint_epoch_*.pt"))
            if checkpoints:
                checkpoint_to_load = checkpoints[-1]
                print(f"\n{'='*60}")
                print(f"RESUMING FROM CHECKPOINT")
                print(f"{'='*60}")
                print(f"Checkpoint: {checkpoint_to_load}")
                print(f"{'='*60}\n")
            else:
                print(f"No checkpoints found in {checkpoint_path}")
        elif checkpoint_path.is_file():
            checkpoint_to_load = checkpoint_path
            print(f"\n{'='*60}")
            print(f"RESUMING FROM CHECKPOINT FILE")
            print(f"{'='*60}")
            print(f"Checkpoint: {checkpoint_to_load}") 
        else:
            print(f"Checkpoint file not found: {checkpoint_path}")

    # Prepare for distributed training
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    # Total optimizer update steps across all epochs — drives the VJEPA
    # teacher-forcing schedule (Bernoulli p = 1 -> 0 over training).
    total_update_steps = max(
        1,
        math.ceil((len(dataloader) * num_epochs) / max(gradient_accumulation_steps, 1)),
    )
    print(
        f"Teacher-force schedule: p=1 for first {tf_warmup_frac:.0%} of "
        f"{total_update_steps} update steps, linear 1->0 over next "
        f"{tf_anneal_frac:.0%}, p=0 afterwards."
    )

    if checkpoint_to_load is not None:
        print(f"Loading checkpoint from {checkpoint_to_load}...")
        checkpoint = torch.load(checkpoint_to_load, map_location="cpu")
        
        # Load model state
        unwrapped_model = accelerator.unwrap_model(model)

        try:
            unwrapped_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            print("✓ Model state loaded successfully")
        except Exception as e:
            print(f"Warning loading model state: {e}")
            print("Attempting to load with key filtering...")

            model_state = unwrapped_model.state_dict()
            checkpoint_state = checkpoint['model_state_dict']

            filtered_state = {}
            for key in checkpoint_state:
                if key in model_state:
                    if checkpoint_state[key].shape == model_state[key].shape:
                        filtered_state[key] = checkpoint_state[key]
                    else:
                        print(f"  Skipping {key}: shape mismatch {checkpoint_state[key].shape} vs {model_state[key].shape}")
                else:
                    print(f"  Skipping {key}: not in current model")
            
            unwrapped_model.load_state_dict(filtered_state, strict=False)
            print(f"✓ Loaded {len(filtered_state)}/{len(checkpoint_state)} matching parameters")
        
        # Load optimizer state
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("✓ Optimizer state loaded successfully")
        except Exception as e:
            print(f"Warning loading optimizer state: {e}")
            print("Continuing with fresh optimizer state...")


        start_epoch = checkpoint['epoch']
        global_step_offset = checkpoint.get('global_step', 0)

        print(f"✓ Resuming from epoch {start_epoch + 1}")
        print(f"✓ Global step: {global_step_offset}\n")

    # Training loop
    print(f"Starting training from epoch {start_epoch + 1} to {num_epochs}...\n")
    global_step = global_step_offset

    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_noise_loss = 0.0
        epoch_vjepa_loss = 0.0
        
        progress_bar = tqdm(
            dataloader,
            desc=f"Epoch {epoch+1}/{num_epochs}",
            disable=not accelerator.is_local_main_process,
        )
        
        for _, batch in enumerate(progress_bar):
            with accelerator.accumulate(model):
                latents = batch['latents']  # [B, 16, 4, 32, 32]
                vjepa_gt = batch['vjepa_tokens']  # [B, 2048, 1408] - ground truth
                text_embeddings = batch['text_embeddings']  # [B, seq_len, 4096]
                
                batch_size = latents.shape[0]
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (batch_size,),
                    device=latents.device,
                ).long()
                
                # Add noise
                noise = torch.randn_like(latents)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                tf_prob = teacher_force_prob_at_step(
                    global_step, total_update_steps, tf_warmup_frac, tf_anneal_frac
                )

                # Forward pass (PredictorP predicts VJEPA, model uses it)
                model_output, predicted_vjepa = model(
                    noisy_latents,
                    timestep=timesteps,
                    encoder_hidden_states=text_embeddings,
                    ground_truth_vjepa=vjepa_gt,
                    teacher_force_prob=tf_prob,
                    enable_temporal_attentions=True,
                )
                
                # Loss 1: Noise prediction (main diffusion loss)
                noise_loss = F.mse_loss(model_output.sample.to(torch.bfloat16), noise.to(torch.bfloat16), reduction="mean")
                
                # Loss 2: VJEPA prediction (PredictorP supervised by ground truth)
                vjepa_loss = F.mse_loss(predicted_vjepa.to(torch.bfloat16), vjepa_gt.to(torch.bfloat16), reduction="mean")
                
                # Combined loss
                total_loss = noise_loss + vjepa_loss_weight * vjepa_loss
                
                # Backprop
                accelerator.backward(total_loss)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad],
                        max_norm=1.0
                    )
                
                optimizer.step()
                optimizer.zero_grad()
                
                if accelerator.sync_gradients:
                    global_step += 1
                
                # Logging
                epoch_loss += total_loss.detach().item()
                epoch_noise_loss += noise_loss.detach().item()
                epoch_vjepa_loss += vjepa_loss.detach().item()
                
                progress_bar.set_postfix({
                    "loss": total_loss.detach().item(),
                    "noise": noise_loss.detach().item(),
                    "vjepa": vjepa_loss.detach().item(),
                    "tf_p": tf_prob,
                })
        
        # Epoch summary
        avg_loss = epoch_loss / len(dataloader)
        avg_noise = epoch_noise_loss / len(dataloader)
        avg_vjepa = epoch_vjepa_loss / len(dataloader)
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Total loss:  {avg_loss:.4f}")
        print(f"  Noise loss:  {avg_noise:.4f}")
        print(f"  VJEPA loss:  {avg_vjepa:.4f}")
        
        # Save checkpoint
        if accelerator.is_main_process:
            checkpoint_path = output_dir / f"checkpoint_epoch_{epoch+1}.pt"
            
            unwrapped_model = accelerator.unwrap_model(model)
            torch.save({
                'epoch': epoch + 1,
                'global_step': global_step,
                'model_state_dict': unwrapped_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'noise_loss': avg_noise,
                'vjepa_loss': avg_vjepa,
            }, checkpoint_path)
            
            print(f"✓ Saved: {checkpoint_path}\n")
    
    print("✓ Training complete!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_json", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./latte_predictor_checkpoints")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Accumulate gradients over N steps. Effective batch = batch_size * gradient_accumulation_steps")

    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--vjepa_weight", type=float, default=1.0, help="VJEPA loss weight")
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--mixed_precision", type=str, default="bf16")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--tf_warmup_frac", type=float, default=0.25,
                        help="Fraction of training with teacher forcing prob=1.0")
    parser.add_argument("--tf_anneal_frac", type=float, default=0.5,
                        help="Fraction of training over which teacher forcing prob anneals 1.0 -> 0.0")

    args = parser.parse_args()

    train_with_predictor(
        index_json_path=args.index_json,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        vjepa_loss_weight=args.vjepa_weight,
        num_train_samples=args.num_samples,
        mixed_precision=args.mixed_precision,
        resume_from_checkpoint=args.resume_from_checkpoint,
        tf_warmup_frac=args.tf_warmup_frac,
        tf_anneal_frac=args.tf_anneal_frac,
    )