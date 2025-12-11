"""
Physics-Informed Video Diffusion Model Training with CogVideoX
Complete implementation with LoRA adapters and physics cross-attention
NOW WITH WEIGHTS & BIASES INTEGRATION
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
from typing import Optional
import logging
from pathlib import Path
import json
import wandb

from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed

from functools import partial
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from accelerate import FullyShardedDataParallelPlugin
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp import CPUOffload

# ============================================================================
# CONFIG
# ============================================================================
class Config:
    BATCH_SIZE = 1
    T_STEPS = 1000
    LAMBDA_PRED = 0.1
    ALPHA_INIT = 0.02
    ADAPTER_LR = 1e-3
    P_LR = 1e-4
    WEIGHT_DECAY = 1e-2
    EPOCHS = 30
    SAVE_EVERY = 500
    LOG_EVERY = 10
    MAX_GRAD_NORM = 1.0
    
    # Checkpoint resuming
    RESUME_FROM_CHECKPOINT = None
    
    # LoRA Config
    LORA_RANK = 64
    LORA_ALPHA = 64
    LORA_DROPOUT = 0.0
    LORA_TARGET_MODULES = [
        "to_q", "to_k", "to_v", "to_out.0",
        "ff.net.0.proj", "ff.net.2"
    ]
    
    # Model dimensions
    VFM_SEQ_LEN = 6144
    VFM_DIM = 1408
    TEXT_SEQ_LEN = 226
    TEXT_DIM = 4096 #  t5-v1_1-xxl hidden size
    
    # VAE latent shape
    LATENT_C = 16
    LATENT_T = 13
    LATENT_H = 60
    LATENT_W = 90
    
    DIM_T = 256
    HIDDEN_DIM = 4096 
    
    # Model
    MODEL_NAME = "THUDM/CogVideoX-2b"
    
    # Paths
    CHECKPOINT_DIR = Path('/scratch/sk12590/PhysVideoGenerator/checkpoints')
    LOG_DIR = Path('/scratch/sk12590/PhysVideoGenerator/logs')

    DATASET_INDEX = '/scratch/sk12590/PhysVideoGenerator/data/indexed_dataset.json'
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    USE_8BIT_ADAM = True
    
    # ========== WANDB CONFIG (ADDED) ==========
    WANDB_PROJECT = "physics-video-diffusion"
    WANDB_ENTITY = "sk12590"  # Set to your wandb username/team if needed
    WANDB_RUN_NAME = "CV_PhysVideoGenerator"  # Auto-generated if None
    WANDB_MODE = "offline"  # "online", "offline", or "disabled"
    WANDB_LOG_INTERVAL = 10  # Log every N steps
    WANDB_SAVE_CODE = False  # Save this script to wandb
    WANDB_API_KEY = "61b31217f6c3fee5fd570133a1a777e3bf035c7c"

# ============================================================================
# CHECK DEPENDENCIES
# ============================================================================
try:
    from diffusers import CogVideoXTransformer3DModel
    from diffusers.models.modeling_outputs import Transformer2DModelOutput
    from peft import LoraConfig, get_peft_model
    DEPS_AVAILABLE = True
except ImportError as e:
    DEPS_AVAILABLE = False
    print(f"Missing dependencies: {e}")
    print("Install: pip install diffusers peft transformers accelerate")

# ============================================================================
# DIFFUSION SCHEDULE
# ============================================================================
def make_beta_schedule(T_steps: int, schedule_type: str = 'linear'):
    if schedule_type == 'linear':
        beta_start = 1e-4
        beta_end = 0.02
        betas = torch.linspace(beta_start, beta_end, T_steps)
    elif schedule_type == 'cosine':
        s = 0.008
        steps = T_steps + 1
        x = torch.linspace(0, T_steps, steps)
        alphas_cumprod = torch.cos(((x / T_steps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = torch.clip(betas, 0.0001, 0.9999)
    
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    return betas, alphas, alpha_bar

def q_sample(z0: torch.Tensor, t: torch.Tensor, alpha_bar: torch.Tensor, 
             noise: Optional[torch.Tensor] = None):
    if noise is None:
        noise = torch.randn_like(z0)
    
    alpha_bar_t = alpha_bar[t].view(-1, 1, 1, 1, 1)
    z_t = torch.sqrt(alpha_bar_t) * z0 + torch.sqrt(1 - alpha_bar_t) * noise

    z_t = z_t.contiguous()
    noise = noise.contiguous()
    return z_t, noise

def sinusoidal_timestep_embedding(t: torch.Tensor, dim: int):
    half_dim = dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=torch.float16) * -emb)
    emb = t.half()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    emb = emb.half()
    return emb

# ============================================================================
# PHYSICS CROSS-ATTENTION MODULE
# ============================================================================
class PhysicsCrossAttention(nn.Module):
    """Physics-informed cross-attention with gating."""
    def __init__(self, query_dim: int, context_dim: int, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.to_q = nn.Linear(query_dim, query_dim, bias=False)
        self.to_k = nn.Linear(context_dim, query_dim, bias=False)
        self.to_v = nn.Linear(context_dim, query_dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(query_dim, query_dim),
            nn.Dropout(dropout)
        )
        
        self.alpha = nn.Parameter(torch.tensor([0.02]))
        
    def forward(self, hidden_states: torch.Tensor, physics_context: torch.Tensor):
        B, N, D = hidden_states.shape
        L = physics_context.shape[1]
        
        q = self.to_q(hidden_states).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.to_k(physics_context).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.to_v(physics_context).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        out = self.to_out(out)
        
        return out * self.alpha

# ============================================================================
# PREDICTOR P
# ============================================================================
class PredictorP(nn.Module):
    """Predicts VFM tokens from (z_t, text_tokens, t_emb)."""
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        latent_dim = 512
        self.latent_encoder = nn.Sequential(
            nn.Conv3d(config.LATENT_C, latent_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, latent_dim),
            nn.SiLU(),
            nn.Conv3d(latent_dim, latent_dim, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, latent_dim),
            nn.SiLU(),
        )
        
        # Calculate positional embedding size dynamically based on config
        # After stride=2 convolution: T/2, H/2, W/2
        latent_t_out = (config.LATENT_T - 1) // 2 + 1
        latent_h_out = (config.LATENT_H - 1) // 2 + 1
        latent_w_out = (config.LATENT_W - 1) // 2 + 1
        pos_embed_seq_len = latent_t_out * latent_h_out * latent_w_out
        self.latent_pos_embed = nn.Parameter(torch.randn(1, pos_embed_seq_len, latent_dim) * 0.02)
        self.latent_to_hidden = nn.Linear(latent_dim, config.HIDDEN_DIM)
        
        self.text_proj = nn.Sequential(
            nn.Linear(config.TEXT_DIM, config.HIDDEN_DIM),
            nn.LayerNorm(config.HIDDEN_DIM),
            nn.GELU()
        )
        
        self.time_proj = nn.Sequential(
            nn.Linear(config.DIM_T, config.HIDDEN_DIM),
            nn.LayerNorm(config.HIDDEN_DIM),
            nn.GELU()
        )
        
        self.fusion_attn = nn.ModuleList([
            nn.MultiheadAttention(config.HIDDEN_DIM, num_heads=8, batch_first=True)
            for _ in range(2)
        ])
        self.fusion_norms = nn.ModuleList([
            nn.LayerNorm(config.HIDDEN_DIM) for _ in range(2)
        ])
        
        self.vfm_queries = nn.Parameter(torch.randn(1, config.VFM_SEQ_LEN, config.HIDDEN_DIM) * 0.02)
        self.vfm_decoder = nn.ModuleList([
            nn.MultiheadAttention(config.HIDDEN_DIM, num_heads=8, batch_first=True)
            for _ in range(3)
        ])
        self.vfm_norms = nn.ModuleList([
            nn.LayerNorm(config.HIDDEN_DIM) for _ in range(3)
        ])
        self.vfm_ffn = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM * 4),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(config.HIDDEN_DIM * 4, config.HIDDEN_DIM)
            ) for _ in range(3)
        ])
        
        self.vfm_out_proj = nn.Linear(config.HIDDEN_DIM, config.VFM_DIM)
        
    def forward(self, z_t: torch.Tensor, text_tokens: torch.Tensor, t_emb: torch.Tensor):
        """
        Args:
            z_t: [B, C, T, H, W] - noisy latent
            text_tokens: [B, L, D] - text embeddings
            t_emb: [B, DIM_T] - time embeddings
        Returns:
            predicted_vfm: [B, VFM_SEQ_LEN, VFM_DIM]
        """
        B = z_t.size(0)
        
        # Encode latent: [B, C, T, H, W] -> [B, latent_dim, T/2, H/2, W/2]
        z_feat = self.latent_encoder(z_t).flatten(2).transpose(1, 2)  # [B, seq_len, latent_dim]
        
        # Verify shape matches positional embedding
        if z_feat.size(1) != self.latent_pos_embed.size(1):
            raise RuntimeError(
                f"Latent feature sequence length {z_feat.size(1)} doesn't match "
                f"positional embedding size {self.latent_pos_embed.size(1)}. "
                f"Check your config: LATENT_T={self.config.LATENT_T}, "
                f"LATENT_H={self.config.LATENT_H}, LATENT_W={self.config.LATENT_W}"
            )
        
        z_feat = z_feat + self.latent_pos_embed
        z_feat = self.latent_to_hidden(z_feat)
        
        text_feat = self.text_proj(text_tokens)
        time_feat = self.time_proj(t_emb).unsqueeze(1)
        
        context = torch.cat([text_feat, time_feat], dim=1)
        for attn, norm in zip(self.fusion_attn, self.fusion_norms):
            attn_out, _ = attn(z_feat, context, context)
            z_feat = norm(z_feat + attn_out)
        
        vfm_tokens = self.vfm_queries.expand(B, -1, -1)
        for attn, norm, ffn in zip(self.vfm_decoder, self.vfm_norms, self.vfm_ffn):
            attn_out, _ = attn(vfm_tokens, z_feat, z_feat)
            vfm_tokens = norm(vfm_tokens + attn_out)
            vfm_tokens = vfm_tokens + ffn(vfm_tokens)
        
        return self.vfm_out_proj(vfm_tokens)

# ============================================================================
# COGVIDEOX BLOCK WITH PHYSICS ATTENTION
# ============================================================================
class CogVideoXBlockWithPhysics(nn.Module):
    """Modified CogVideoXBlock with physics cross-attention."""
    def __init__(self, original_block: nn.Module, physics_attn: PhysicsCrossAttention):
        super().__init__()
        self.norm1 = original_block.norm1
        self.attn1 = original_block.attn1
        self.norm2 = original_block.norm2
        self.ff = original_block.ff
        self.physics_cross_attn = physics_attn
        
    def forward(self, hidden_states, encoder_hidden_states, temb, 
                image_rotary_emb=None, attention_kwargs=None, physics_context=None):
        text_seq_length = encoder_hidden_states.size(1)
        attention_kwargs = attention_kwargs or {}

        # 1. Self-Attention
        norm_hidden_states, norm_encoder_hidden_states, gate_msa, enc_gate_msa = self.norm1(
            hidden_states, encoder_hidden_states, temb
        )

        attn_hidden_states, attn_encoder_hidden_states = self.attn1(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
            **attention_kwargs,
        )

        hidden_states = hidden_states + gate_msa * attn_hidden_states
        encoder_hidden_states = encoder_hidden_states + enc_gate_msa * attn_encoder_hidden_states

        # 2. Physics Cross-Attention (INSERTED HERE!)
        if physics_context is not None:
            physics_output = self.physics_cross_attn(hidden_states, physics_context)
            hidden_states = hidden_states + physics_output

        # 3. Feed-Forward
        norm_hidden_states, norm_encoder_hidden_states, gate_ff, enc_gate_ff = self.norm2(
            hidden_states, encoder_hidden_states, temb
        )

        norm_hidden_states = torch.cat([norm_encoder_hidden_states, norm_hidden_states], dim=1)
        ff_output = self.ff(norm_hidden_states)

        hidden_states = hidden_states + gate_ff * ff_output[:, text_seq_length:]
        encoder_hidden_states = encoder_hidden_states + enc_gate_ff * ff_output[:, :text_seq_length]

        return hidden_states, encoder_hidden_states

# ============================================================================
# COGVIDEOX WITH LORA AND PHYSICS
# ============================================================================
class CogVideoXWithPhysics(nn.Module):
    """CogVideoX with LoRA and physics cross-attention."""
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        if not DEPS_AVAILABLE:
            raise ImportError("Required packages not installed")
        
        print(f"Loading CogVideoX from {config.MODEL_NAME}...")
        self.transformer = CogVideoXTransformer3DModel.from_pretrained(
            config.MODEL_NAME,
            subfolder="transformer",
            torch_dtype=torch.float16
        )

        self.transformer.set_attn_processor(torch.nn.functional.scaled_dot_product_attention)

        self.transformer.enable_gradient_checkpointing()

        print("Configuring LoRA adapters...")
        lora_config = LoraConfig(
            r=config.LORA_RANK,
            lora_alpha=config.LORA_ALPHA,
            init_lora_weights="gaussian",
            target_modules=config.LORA_TARGET_MODULES,
            lora_dropout=config.LORA_DROPOUT,
        )
        
        self.transformer = get_peft_model(self.transformer, lora_config)
        
        trainable_params = sum(p.numel() for p in self.transformer.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.transformer.parameters())
        print(f"trainable params: {trainable_params:,} || all params: {total_params:,} || trainable%: {100 * trainable_params / total_params:.2f}")
        
        print("Injecting physics cross-attention...")
        self.physics_attns = nn.ModuleList()
        
        original_blocks = self.transformer.transformer_blocks
        new_blocks = nn.ModuleList()
        
        for i, original_block in enumerate(original_blocks):
            if i % 3 == 0:
                hidden_dim = original_block.attn1.to_q.in_features
                physics_attn = PhysicsCrossAttention(
                    query_dim=hidden_dim,
                    context_dim=config.VFM_DIM,
                    num_heads=8
                )
                modified_block = CogVideoXBlockWithPhysics(original_block, physics_attn)
                self.physics_attns.append(physics_attn)
            else:
                modified_block = original_block
            new_blocks.append(modified_block)
            
        self.transformer.to(torch.bfloat16)

        for physics_attn in self.physics_attns:
            physics_attn.to(torch.bfloat16)
        
        self.transformer.transformer_blocks = new_blocks
        print(f"✓ Injected physics attention into {len(self.physics_attns)} blocks")
        
    def forward(self, hidden_states, encoder_hidden_states, timestep, predicted_vfm,
                timestep_cond=None, ofs=None, image_rotary_emb=None, return_dict=True):
        """
        Args:
            hidden_states: [B, C, T, H, W] - noisy video latents in standard PyTorch format
            encoder_hidden_states: [B, L, D] - text embeddings
            timestep: [B] - diffusion timesteps
            predicted_vfm: [B, 6144, 1408] - predicted VFM tokens
        """
        # CRITICAL: Correct shape unpacking for [B, C, T, H, W] format
        batch_size, channels, num_frames, height, width = hidden_states.shape
        
        # Verify input shape matches config
        assert channels == self.config.LATENT_C, f"Expected {self.config.LATENT_C} channels, got {channels}"
        assert num_frames == self.config.LATENT_T, f"Expected {self.config.LATENT_T} frames, got {num_frames}"
        
        # 1. Time embedding
        t_emb = self.transformer.time_proj(timestep)
        t_emb = t_emb.to(dtype=hidden_states.dtype)
        emb = self.transformer.time_embedding(t_emb, timestep_cond)
        
        if self.transformer.ofs_embedding is not None and ofs is not None:
            ofs_emb = self.transformer.ofs_proj(ofs)
            ofs_emb = ofs_emb.to(dtype=hidden_states.dtype)
            ofs_emb = self.transformer.ofs_embedding(ofs_emb)
            emb = emb + ofs_emb
        
        # 2. Patch embedding
        # Note: CogVideoX expects [B, T, C, H, W] format for patch_embed
        hidden_states_reordered = hidden_states.permute(0, 2, 1, 3, 4).contiguous()  # [B, C, T, H, W] -> [B, T, C, H, W]
        hidden_states_patched = self.transformer.patch_embed(encoder_hidden_states, hidden_states_reordered)
        hidden_states_patched = self.transformer.embedding_dropout(hidden_states_patched)
        
        text_seq_length = encoder_hidden_states.shape[1]
        encoder_hidden_states_patched = hidden_states_patched[:, :text_seq_length]
        hidden_states_patched = hidden_states_patched[:, text_seq_length:]
        
        # 3. Transformer blocks with physics attention
        for block in self.transformer.transformer_blocks:
            if isinstance(block, CogVideoXBlockWithPhysics):
                hidden_states_patched, encoder_hidden_states_patched = block(
                    hidden_states=hidden_states_patched,
                    encoder_hidden_states=encoder_hidden_states_patched,
                    temb=emb,
                    image_rotary_emb=image_rotary_emb,
                    attention_kwargs=None,
                    physics_context=predicted_vfm,
                )
            else:
                hidden_states_patched, encoder_hidden_states_patched = block(
                    hidden_states=hidden_states_patched,
                    encoder_hidden_states=encoder_hidden_states_patched,
                    temb=emb,
                    image_rotary_emb=image_rotary_emb,
                    attention_kwargs=None,
                )
        
        # 4. Final normalization
        if not self.transformer.config.use_rotary_positional_embeddings:
            hidden_states_patched = self.transformer.norm_final(hidden_states_patched)
        else:
            hidden_states_patched = torch.cat([encoder_hidden_states_patched, hidden_states_patched], dim=1)
            hidden_states_patched = self.transformer.norm_final(hidden_states_patched)
            hidden_states_patched = hidden_states_patched[:, text_seq_length:]
        
        # 5. Output projection
        hidden_states_patched = self.transformer.norm_out(hidden_states_patched, temb=emb)
        hidden_states_patched = self.transformer.proj_out(hidden_states_patched)
        
        # 6. Unpatchify - uses original num_frames from reordered input
        p = self.transformer.config.patch_size
        p_t = self.transformer.config.patch_size_t
        
        if p_t is None:
            # CogVideoX 1.0 unpatchify
            output = hidden_states_patched.reshape(batch_size, num_frames, height // p, width // p, -1, p, p)
            output = output.permute(0, 1, 4, 2, 5, 3, 6).flatten(5, 6).flatten(3, 4)
        else:
            # CogVideoX 1.5 unpatchify
            output = hidden_states_patched.reshape(
                batch_size, (num_frames + p_t - 1) // p_t, height // p, width // p, -1, p_t, p, p
            )
            output = output.permute(0, 1, 5, 4, 2, 6, 3, 7).flatten(6, 7).flatten(4, 5).flatten(1, 2)
        
        # Convert back to [B, C, T, H, W] format
        output = output.permute(0, 2, 1, 3, 4).contiguous()  # [B, T, C, H, W] -> [B, C, T, H, W]
        
        if not return_dict:
            return (output,)
        
        return Transformer2DModelOutput(sample=output)

    def get_trainable_parameters(self):
        return [p for p in self.transformer.parameters() if p.requires_grad]

# ============================================================================
# DATASET
# ============================================================================
class VideoPhysicsDataset(Dataset):
    """Dataset loading from index.json."""
    def __init__(self, index_json_path: str, config: Config, cache_in_memory: bool = False):
        self.config = config
        self.cache_in_memory = cache_in_memory
        
        with open(index_json_path, 'r') as f:
            self.data_index = json.load(f)
        
        print(f"Loaded {len(self.data_index)} samples from {index_json_path}")
        
        if cache_in_memory:
            print("Caching data in memory...")
            self.cached_data = [self._load_sample(i) for i in range(len(self.data_index))]
        else:
            self.cached_data = None
        
        self._verify_first_sample()
    
    def _load_sample(self, idx):
        sample_info = self.data_index[idx]
        
        vae_npz = np.load(sample_info['vae'])
        if 'arr_0' in vae_npz:
            z0 = vae_npz['arr_0']
        elif 'frames' in vae_npz:
            z0 = vae_npz['latents']
        else:
            z0 = vae_npz[vae_npz.files[0]]
        z0 = torch.from_numpy(z0).squeeze(0).to(torch.float16)

        vjepa_npz = np.load(sample_info['vjepa'])
        if 'arr_0' in vae_npz:
            vfm_tokens = vjepa_npz['arr_0']
        elif 'frames' in vae_npz:
            vfm_tokens = vjepa_npz['latents']
        else:
            vfm_tokens = vjepa_npz[vjepa_npz.files[0]]
        vfm_tokens = torch.from_numpy(vfm_tokens).squeeze(0).to(torch.float16)

        text_tokens = torch.from_numpy(np.load(sample_info['text'])).squeeze(0).to(torch.float16)
        return z0, vfm_tokens, text_tokens
    
    def _verify_first_sample(self):
        z0, vfm_tokens, text_tokens = self._load_sample(0)
        print(f"Sample shapes: VAE={list(z0.shape)}, VFM={list(vfm_tokens.shape)}, Text={list(text_tokens.shape)}")
    
    def __len__(self):
        return len(self.data_index)
    
    def __getitem__(self, idx):
        if self.cached_data is not None:
            return self.cached_data[idx]
        return self._load_sample(idx)

# ============================================================================
# CHECKPOINT UTILITIES
# ============================================================================
def save_checkpoint(vdm, predictor, optimizer, epoch, step, config):
    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    
    lora_path = config.CHECKPOINT_DIR / f'lora_epoch{epoch}_step{step}'
    vdm.transformer.save_pretrained(lora_path)
    
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'physics_attns': [attn.state_dict() for attn in vdm.physics_attns],
        'predictor': predictor.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    
    path = config.CHECKPOINT_DIR / f'checkpoint_epoch{epoch}_step{step}.pt'
    torch.save(checkpoint, path)
    logging.info(f"Saved checkpoint to {path}")
    
    # ========== WANDB: Save checkpoint (ADDED) ==========
    if wandb.run is not None:
        wandb.save(str(path))
        wandb.save(str(lora_path / '*'))

def load_checkpoint(vdm, predictor, optimizer, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    epoch = checkpoint['epoch']
    step = checkpoint['step']
    
    lora_dir_name = Path(checkpoint_path).stem.replace('checkpoint_', 'lora_')
    lora_path = Path(checkpoint_path).parent / lora_dir_name
    
    if lora_path.exists():
        from peft import PeftModel
        vdm.transformer = PeftModel.from_pretrained(
            vdm.transformer.base_model.model,
            lora_path,
            is_trainable=True
        )
    
    for i, state_dict in enumerate(checkpoint['physics_attns']):
        vdm.physics_attns[i].load_state_dict(state_dict)
    
    predictor.load_state_dict(checkpoint['predictor'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    
    logging.info(f"✓ Resumed from epoch {epoch}, step {step}")
    return epoch, step

# ============================================================================
# TRAINING
# ============================================================================
def train(config: Config):
    fsdp_plugin = FullyShardedDataParallelPlugin(
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision_policy=None, # Accelerate handles this from config
        auto_wrap_policy=partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={CogVideoXBlockWithPhysics}, # <--- The Magic Fix
        ),
        cpu_offload=CPUOffload(offload_params=True),
    )

    accelerator = Accelerator(
        mixed_precision="bf16",
        log_with="wandb",
        project_dir=config.LOG_DIR,
        fsdp_plugin=fsdp_plugin
    )

    if accelerator.is_main_process:
        config.LOG_DIR.mkdir(parents=True, exist_ok=True)
        config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            handlers=[logging.FileHandler(config.LOG_DIR / 'training.log'), logging.StreamHandler()]
        )

        # Initialize trackers (replaces wandb.init)
        accelerator.init_trackers(
            project_name=config.WANDB_PROJECT,
            config={k:str(v) for k,v in config.__dict__.items() if not k.startswith('__')}
        )

    # device = torch.device(config.DEVICE)

    vdm = CogVideoXWithPhysics(config)
    vdm = vdm.to(torch.bfloat16)

    predictor = PredictorP(config)
    predictor = predictor.to(torch.bfloat16)

    trainable_params = vdm.get_trainable_parameters() + list(predictor.parameters())

    optimizer = AdamW(trainable_params, lr=config.ADAPTER_LR, weight_decay=config.WEIGHT_DECAY)
    
    dataset = VideoPhysicsDataset(config.DATASET_INDEX, config, cache_in_memory=False)
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True, 
                          num_workers=4, pin_memory=True)
    
    vdm, predictor, optimizer, dataloader = accelerator.prepare(
        vdm, predictor, optimizer, dataloader
    )

    betas, alphas, alpha_bar = make_beta_schedule(config.T_STEPS)
    alpha_bar = alpha_bar.to(accelerator.device)
    
    # global_step = start_step
    global_step = 0
    start_epoch = 0
    
    for epoch in range(start_epoch, config.EPOCHS):
        vdm.train()
        predictor.train()
        
        for batch in dataloader:
            z0, vfm_tokens, text_tokens = batch

            B = z0.size(0)
            t = torch.randint(0, config.T_STEPS, (B,), device=accelerator.device)
            z_t, noise = q_sample(z0, t, alpha_bar)

            t_emb = sinusoidal_timestep_embedding(t, config.DIM_T)

            # Forward passes
            predicted_vfm = predictor(z_t, text_tokens, t_emb)
            eps_hat = vdm(
                hidden_states=z_t,
                encoder_hidden_states=text_tokens,
                timestep=t,
                predicted_vfm=predicted_vfm
            ).sample
            
            L_diff = F.mse_loss(eps_hat, noise)
            L_pred = F.mse_loss(predicted_vfm, vfm_tokens)
            loss = L_diff + config.LAMBDA_PRED * L_pred
            
            # ### CHANGED: Backprop using accelerator
            accelerator.backward(loss)
            
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(trainable_params, config.MAX_GRAD_NORM)
            
            optimizer.step()
            optimizer.zero_grad()
            
            # Logging (only on main process)
            if accelerator.is_main_process and global_step % config.WANDB_LOG_INTERVAL == 0:
                accelerator.log({
                    "train/loss": loss.item(),
                    "train/l_diff": L_diff.item(),
                    "train/l_pred": L_pred.item(),
                }, step=global_step)
                
                logging.info(f"Step {global_step} Loss: {loss.item():.4f}")
            
            # ### CHANGED: Saving Checkpoints with FSDP
            if global_step % config.SAVE_EVERY == 0 and global_step > 0:
                accelerator.wait_for_everyone()
                # Unwrap models to save clean state dicts
                unwrapped_vdm = accelerator.unwrap_model(vdm)
                unwrapped_predictor = accelerator.unwrap_model(predictor)
                
                if accelerator.is_main_process:
                    save_path = config.CHECKPOINT_DIR / f'checkpoint_{global_step}.pt'
                    torch.save({
                        'vdm': unwrapped_vdm.state_dict(),
                        'predictor': unwrapped_predictor.state_dict(),
                    }, save_path)
                    logging.info(f"Saved to {save_path}")
            
            global_step += 1
    
    logging.info("Training complete!")
    
    # ========== WANDB: Finish (ADDED) ==========
    # wandb.finish()

if __name__ == '__main__':
    config = Config()
    train(config)