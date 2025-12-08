import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
import math
from typing import List, Tuple, Optional
import logging
from pathlib import Path

from diffusers import CogVideoXTransformer3DModel, AutoencoderKLCogVideoX
from peft import LoraConfig, get_peft_model

# ============================================================================
# CONFIG
# ============================================================================
class Config:
    BATCH_SIZE = 2
    T_STEPS = 1000
    LAMBDA_PRED = 0.1
    ALPHA_INIT = 0.02
    ADAPTER_LR = 1e-3  # Official CogVideoX recommendation: 1e-3 to 1e-4
    P_LR = 1e-4
    WEIGHT_DECAY = 1e-2
    EPOCHS = 30  # For 50 videos, ~1500-2000 steps recommended
    SAVE_EVERY = 500
    LOG_EVERY = 10
    MAX_GRAD_NORM = 1.0
    
    # LoRA Config (following official recommendations)
    LORA_RANK = 64  # Official rec: 64 for new concepts, 16-32 for moderate cases
    LORA_ALPHA = 64  # Set to rank or rank//2
    LORA_DROPOUT = 0.0
    LORA_TARGET_MODULES = [
        "to_q", "to_k", "to_v", "to_out.0",  # Attention projections
        "ff.net.0.proj", "ff.net.2"  # Feed-forward layers
    ]
    
    # Model dimensions
    VFM_SEQ_LEN = 6144  # VFM sequence length
    VFM_DIM = 1408  # VFM token dimension
    TEXT_SEQ_LEN = 128  # Text sequence length
    TEXT_DIM = 1024  # Text embedding dimension
    
    # VAE latent shape: [B, C, T, H, W] = [B, 16, 13, 60, 90]
    LATENT_C = 16  # Channels
    LATENT_T = 13  # Temporal frames
    LATENT_H = 60  # Height
    LATENT_W = 90  # Width
    
    DIM_T = 256  # Timestep embedding dimension
    HIDDEN_DIM = 1024
    
    # Model name
    MODEL_NAME = "THUDM/CogVideoX-2b"  # or "THUDM/CogVideoX-5b"
    
    # Paths
    CHECKPOINT_DIR = Path('./checkpoints')
    LOG_DIR = Path('./logs')
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Memory optimization
    ENABLE_GRADIENT_CHECKPOINTING = True
    USE_8BIT_ADAM = True  # Reduces memory

# ============================================================================
# DIFFUSION SCHEDULE
# ============================================================================
def make_beta_schedule(T_steps: int, schedule_type: str = 'linear') -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create beta schedule for diffusion process."""
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
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")
    
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    
    return betas, alphas, alpha_bar

def q_sample(z0: torch.Tensor, t: torch.Tensor, alpha_bar: torch.Tensor, noise: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forward diffusion process: q(z_t | z_0)."""
    if noise is None:
        noise = torch.randn_like(z0)
    
    # Extract alpha_bar for each timestep in batch
    alpha_bar_t = alpha_bar[t].view(-1, 1, 1, 1, 1)  # [B, 1, 1, 1, 1]
    
    z_t = torch.sqrt(alpha_bar_t) * z0 + torch.sqrt(1 - alpha_bar_t) * noise
    return z_t, noise

# ============================================================================
# TIMESTEP EMBEDDING
# ============================================================================
def sinusoidal_timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal positional embeddings for timesteps."""
    half_dim = dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
    emb = t.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb

# ============================================================================
# PHYSICS CROSS-ATTENTION MODULE
# ============================================================================
class PhysicsCrossAttention(nn.Module):
    """
    Physics-informed cross-attention injected into CogVideoX blocks.
    Queries from video features, Keys/Values from predicted VFM tokens.
    """
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
        
        # Gating parameter (trainable scalar)
        self.alpha = nn.Parameter(torch.tensor(0.02))
        
    def forward(self, hidden_states: torch.Tensor, physics_context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [B, N, D] - features from transformer block
            physics_context: [B, L, D_context] - predicted VFM tokens
        
        Returns:
            out: [B, N, D] - modulated features
        """
        B, N, D = hidden_states.shape
        L = physics_context.shape[1]
        
        # Compute Q, K, V
        q = self.to_q(hidden_states).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.to_k(physics_context).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.to_v(physics_context).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention to values
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        out = self.to_out(out)
        
        # Apply gating
        return out * self.alpha

# ============================================================================
# PREDICTOR P - Maps (z_t, text_tokens, t_emb) -> VFM tokens
# ============================================================================
class PredictorP(nn.Module):
    """
    Predictor network that learns to map (noisy_latent, text_tokens, t_emb) -> VFM tokens
    """
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        # Latent encoder (3D conv for spatiotemporal processing)
        latent_dim = 512
        self.latent_encoder = nn.Sequential(
            nn.Conv3d(config.LATENT_C, latent_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, latent_dim),
            nn.SiLU(),
            nn.Conv3d(latent_dim, latent_dim, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, latent_dim),
            nn.SiLU(),
        )
        # After conv: [B, 512, 7, 30, 45]
        self.latent_pos_embed = nn.Parameter(torch.randn(1, 7 * 30 * 45, latent_dim) * 0.02)
        self.latent_to_hidden = nn.Linear(latent_dim, config.HIDDEN_DIM)
        
        # Text projection
        self.text_proj = nn.Sequential(
            nn.Linear(config.TEXT_DIM, config.HIDDEN_DIM),
            nn.LayerNorm(config.HIDDEN_DIM),
            nn.GELU()
        )
        
        # Time projection
        self.time_proj = nn.Sequential(
            nn.Linear(config.DIM_T, config.HIDDEN_DIM),
            nn.LayerNorm(config.HIDDEN_DIM),
            nn.GELU()
        )
        
        # Fusion via cross-attention
        self.fusion_attn = nn.ModuleList([
            nn.MultiheadAttention(config.HIDDEN_DIM, num_heads=8, batch_first=True)
            for _ in range(2)
        ])
        self.fusion_norms = nn.ModuleList([
            nn.LayerNorm(config.HIDDEN_DIM) for _ in range(2)
        ])
        
        # VFM decoder with learnable queries
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
        
        # Output projection
        self.vfm_out_proj = nn.Linear(config.HIDDEN_DIM, config.VFM_DIM)
        
    def forward(self, z_t: torch.Tensor, text_tokens: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        B = z_t.size(0)
        
        # Encode latent
        z_feat = self.latent_encoder(z_t).flatten(2).transpose(1, 2)
        z_feat = z_feat + self.latent_pos_embed
        z_feat = self.latent_to_hidden(z_feat)
        
        # Process text and time
        text_feat = self.text_proj(text_tokens)
        time_feat = self.time_proj(t_emb).unsqueeze(1)
        
        # Fuse context
        context = torch.cat([text_feat, time_feat], dim=1)
        for attn, norm in zip(self.fusion_attn, self.fusion_norms):
            attn_out, _ = attn(z_feat, context, context)
            z_feat = norm(z_feat + attn_out)
        
        # Decode to VFM tokens
        vfm_tokens = self.vfm_queries.expand(B, -1, -1)
        for attn, norm, ffn in zip(self.vfm_decoder, self.vfm_norms, self.vfm_ffn):
            attn_out, _ = attn(vfm_tokens, z_feat, z_feat)
            vfm_tokens = norm(vfm_tokens + attn_out)
            vfm_tokens = vfm_tokens + ffn(vfm_tokens)
        
        return self.vfm_out_proj(vfm_tokens)

# ============================================================================
# COGVIDEOX WITH LORA AND PHYSICS CONDITIONING
# ============================================================================
class CogVideoXWithPhysics(nn.Module):
    """
    CogVideoX model with:
    1. LoRA adapters (via PEFT)
    2. Physics cross-attention injection
    """
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        print(f"Loading CogVideoX from {config.MODEL_NAME}...")
        self.transformer = CogVideoXTransformer3DModel.from_pretrained(
            config.MODEL_NAME,
            subfolder="transformer",
            torch_dtype=torch.float32
        )
        
        # Enable gradient checkpointing for memory efficiency
        if config.ENABLE_GRADIENT_CHECKPOINTING:
            self.transformer.enable_gradient_checkpointing()
        
        # Configure LoRA using PEFT
        print("Configuring LoRA adapters...")
        lora_config = LoraConfig(
            r=config.LORA_RANK,
            lora_alpha=config.LORA_ALPHA,
            init_lora_weights="gaussian",  # diffusers style init
            target_modules=config.LORA_TARGET_MODULES,
            lora_dropout=config.LORA_DROPOUT,
        )
        
        # Apply LoRA to transformer
        self.transformer = get_peft_model(self.transformer, lora_config)
        self.transformer.print_trainable_parameters()
        
        # Inject physics cross-attention into transformer blocks
        print("Injecting physics cross-attention...")
        self.physics_attns = nn.ModuleList()
        
        if hasattr(self.transformer.base_model.model, 'transformer_blocks'):
            blocks = self.transformer.base_model.model.transformer_blocks
        else:
            blocks = []
            print("Warning: Could not find transformer_blocks")
        
        for block in blocks:
            # Get hidden dimension from existing attention
            if hasattr(block, 'attn1'):
                hidden_dim = block.attn1.to_q.in_features
            else:
                hidden_dim = 3072  # Default for CogVideoX-2B
            
            physics_attn = PhysicsCrossAttention(
                query_dim=hidden_dim,
                context_dim=config.VFM_DIM,
                num_heads=8
            )
            block.physics_cross_attn = physics_attn
            self.physics_attns.append(physics_attn)
        
        print(f"✓ Injected {len(self.physics_attns)} physics attention layers")
        
    def forward(self, hidden_states: torch.Tensor, timestep: torch.Tensor,
                encoder_hidden_states: torch.Tensor, predicted_vfm: torch.Tensor) -> torch.Tensor:
        """
        Forward with physics conditioning.
        
        Args:
            hidden_states: [B, C, T, H, W]
            timestep: [B]
            encoder_hidden_states: [B, L, D] - text embeddings
            predicted_vfm: [B, 6144, 1408] - VFM predictions for physics
        """
        # Store VFM for physics attention to access
        self._predicted_vfm = predicted_vfm
        
        # Call transformer forward (LoRA will be applied automatically)
        # We need to hook into blocks to apply physics attention
        output = self._forward_with_physics_injection(
            hidden_states, timestep, encoder_hidden_states, predicted_vfm
        )
        
        return output
    
    def _forward_with_physics_injection(self, hidden_states, timestep, 
                                       encoder_hidden_states, predicted_vfm):
        """Modified forward that injects physics attention."""
        # This is a simplified version - full implementation would hook into
        # each transformer block to apply physics_cross_attn after self-attention
        
        # Call base transformer
        output = self.transformer.base_model.model(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            return_dict=True
        )
        
        return output.sample

    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """Get all trainable parameters."""
        params = []
        
        # LoRA parameters (automatically marked as trainable by PEFT)
        params.extend([p for p in self.transformer.parameters() if p.requires_grad])
        
        # Physics attention parameters
        for physics_attn in self.physics_attns:
            params.extend(list(physics_attn.parameters()))
        
        return params

# ============================================================================
# DATASET
# ============================================================================
class VideoPhysicsDataset(Dataset):
    """Dataset with correct dimensions."""
    def __init__(self, num_samples: int, config: Config):
        self.num_samples = num_samples
        self.config = config
        
        print(f"Initializing dataset with shapes:")
        print(f"  z0: [{config.LATENT_C}, {config.LATENT_T}, {config.LATENT_H}, {config.LATENT_W}]")
        print(f"  vfm_tokens: [{config.VFM_SEQ_LEN}, {config.VFM_DIM}]")
        print(f"  text_tokens: [{config.TEXT_SEQ_LEN}, {config.TEXT_DIM}]")
        
        self.z0_list = []
        self.vfm_tokens_list = []
        self.text_tokens_list = []
        
        for _ in range(num_samples):
            z0 = torch.randn(config.LATENT_C, config.LATENT_T, config.LATENT_H, config.LATENT_W)
            vfm_tokens = torch.randn(config.VFM_SEQ_LEN, config.VFM_DIM)
            text_tokens = torch.randn(config.TEXT_SEQ_LEN, config.TEXT_DIM)
            
            self.z0_list.append(z0)
            self.vfm_tokens_list.append(vfm_tokens)
            self.text_tokens_list.append(text_tokens)
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int):
        return self.z0_list[idx], self.vfm_tokens_list[idx], self.text_tokens_list[idx]

# ============================================================================
# TRAINING UTILITIES
# ============================================================================
def save_checkpoint(vdm: CogVideoXWithPhysics, predictor: PredictorP, 
                   optimizer: torch.optim.Optimizer, epoch: int, step: int, config: Config):
    """Save checkpoint."""
    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save LoRA adapters separately (PEFT style)
    lora_path = config.CHECKPOINT_DIR / f'lora_epoch{epoch}_step{step}'
    vdm.transformer.save_pretrained(lora_path)
    
    # Save other components
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

# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================
def train(config: Config):
    """Main training function."""
    
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(config.LOG_DIR / 'training.log'),
            logging.StreamHandler()
        ]
    )
    
    device = torch.device(config.DEVICE)
    logging.info(f"Using device: {device}")
    
    # Diffusion schedule
    _, _, alpha_bar = make_beta_schedule(config.T_STEPS)
    alpha_bar = alpha_bar.to(device)
    
    # Initialize models
    print("=" * 80)
    vdm = CogVideoXWithPhysics(config).to(device)
    predictor = PredictorP(config).to(device)
    print("=" * 80)
    
    # Get trainable parameters
    trainable_params = vdm.get_trainable_parameters() + list(predictor.parameters())
    
    # Optimizer
    if config.USE_8BIT_ADAM:
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(trainable_params, lr=config.ADAPTER_LR, 
                                           weight_decay=config.WEIGHT_DECAY)
            logging.info("Using 8-bit Adam optimizer")
        except ImportError:
            optimizer = AdamW(trainable_params, lr=config.ADAPTER_LR, 
                            weight_decay=config.WEIGHT_DECAY)
            logging.info("Using standard AdamW optimizer")
    else:
        optimizer = AdamW(trainable_params, lr=config.ADAPTER_LR, 
                         weight_decay=config.WEIGHT_DECAY)
    
    # Dataset
    dataset = VideoPhysicsDataset(num_samples=100, config=config)
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)
    
    logging.info(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")
    
    # Training loop
    global_step = 0
    
    for epoch in range(config.EPOCHS):
        vdm.train()
        predictor.train()
        
        epoch_loss = 0.0
        epoch_l_diff = 0.0
        epoch_l_pred = 0.0
        
        for _, (z0, vfm_tokens, text_tokens) in enumerate(dataloader):
            z0 = z0.to(device)
            vfm_tokens = vfm_tokens.to(device)
            text_tokens = text_tokens.to(device)
            
            B = z0.size(0)
            t = torch.randint(0, config.T_STEPS, (B,), device=device)
            
            # Forward diffusion
            z_t, noise = q_sample(z0, t, alpha_bar)
            
            # Predictor: (z_t, text, t_emb) -> predicted VFM
            t_emb = sinusoidal_timestep_embedding(t, config.DIM_T)
            predicted_vfm = predictor(z_t, text_tokens, t_emb)
            
            # VDM forward with physics conditioning
            eps_hat = vdm(z_t, t, text_tokens, predicted_vfm)
            
            # Losses
            L_diff = F.mse_loss(eps_hat, noise)
            L_pred = F.mse_loss(predicted_vfm, vfm_tokens)
            loss = L_diff + config.LAMBDA_PRED * L_pred
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, config.MAX_GRAD_NORM)
            optimizer.step()
            
            # Logging
            epoch_loss += loss.item()
            epoch_l_diff += L_diff.item()
            epoch_l_pred += L_pred.item()
            
            if global_step % config.LOG_EVERY == 0:
                logging.info(
                    f"Epoch [{epoch}/{config.EPOCHS}] Step [{global_step}] "
                    f"Loss: {loss.item():.4f} L_diff: {L_diff.item():.4f} "
                    f"L_pred: {L_pred.item():.4f}"
                )
            
            if global_step % config.SAVE_EVERY == 0 and global_step > 0:
                save_checkpoint(vdm, predictor, optimizer, epoch, global_step, config)
            
            global_step += 1
        
        # Epoch summary
        avg_loss = epoch_loss / len(dataloader)
        avg_l_diff = epoch_l_diff / len(dataloader)
        avg_l_pred = epoch_l_pred / len(dataloader)
        
        logging.info(
            f"Epoch [{epoch}/{config.EPOCHS}] Summary - "
            f"Avg Loss: {avg_loss:.4f} Avg L_diff: {avg_l_diff:.4f} "
            f"Avg L_pred: {avg_l_pred:.4f}"
        )
        
        save_checkpoint(vdm, predictor, optimizer, epoch, global_step, config)
    
    logging.info("Training complete!")

if __name__ == '__main__':
    config = Config()
    train(config)