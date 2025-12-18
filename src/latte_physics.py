# Copyright 2025 - Modified Latte with PredictorP for Physics Conditioning

"""
Complete implementation with PredictorP network.

Architecture:
1. PredictorP: Predicts VJEPA tokens from (noisy_latents, text, timestep)
2. Latte Transformer: Uses predicted VJEPA tokens for physics conditioning in temporal blocks
3. Training: Joint training of PredictorP + temporal cross-attention layers
"""

from typing import Optional
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.attention import BasicTransformerBlock
from diffusers.models.cache_utils import CacheMixin
from diffusers.models.embeddings import PatchEmbed, PixArtAlphaTextProjection, get_1d_sincos_pos_embed_from_grid
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import AdaLayerNormSingle


class PredictorP(nn.Module):
    """
    PredictorP: Predicts VJEPA physics tokens from (noisy_latents, text_embeddings, timestep).
    
    At inference time, we don't have ground-truth VJEPA tokens, so PredictorP predicts them!
    """
    
    def __init__(
        self,
        latent_frames: int = 16,
        latent_channels: int = 4,
        latent_height: int = 32,
        latent_width: int = 32,
        text_dim: int = 4096,
        hidden_dim: int = 512,
        vjepa_seq_len: int = 2048,  # 16 frames: 256 spatial * 8 temporal patches
        vjepa_dim: int = 1408,
        num_layers: int = 4,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.vjepa_seq_len = vjepa_seq_len
        self.vjepa_dim = vjepa_dim
        
        # 1. Latent encoder (process noisy video latents)
        # Input shape: [B, C, T, H, W]
        self.latent_encoder = nn.Sequential(
            nn.Conv3d(latent_channels, hidden_dim // 2, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim // 2),
            nn.SiLU(),
            nn.Conv3d(hidden_dim // 2, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
        )
        # After conv, shape: [B, hidden_dim, T//2, H//2, W//2]
        
        # 2. Text projection
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        
        # 3. Timestep embedding
        self.time_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        
        # 4. Fusion transformer (combine latent + text + time)
        self.fusion_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1,
                activation='gelu',
                batch_first=True,
            )
            for _ in range(num_layers)
        ])
        
        # 5. VJEPA decoder (predict VJEPA tokens)
        self.vjepa_decoder = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1,
                activation='gelu',
                batch_first=True,
            )
            for _ in range(num_layers)
        ])
        
        # Learnable query tokens for VJEPA
        self.vjepa_queries = nn.Parameter(torch.randn(1, vjepa_seq_len, hidden_dim))
        
        # 6. Output projection to VJEPA dimension
        self.output_proj = nn.Linear(hidden_dim, vjepa_dim)
        
    def get_timestep_embedding(self, timesteps, embedding_dim):
        """Sinusoidal timestep embeddings."""
        half_dim = embedding_dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if embedding_dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1))
        return emb
    
    def forward(
        self,
        noisy_latents: torch.Tensor,  # [B, 4, 16, 32, 32]
        text_embeddings: torch.Tensor,  # [B, seq_len, 4096]
        timesteps: torch.Tensor,  # [B]
    ):
        """
        Predict VJEPA tokens from noisy latents, text, and timestep.
        
        Args:
            noisy_latents: [B, C=4, T=16, H=32, W=32]
            text_embeddings: [B, seq_len, 4096]
            timesteps: [B]
        
        Returns:
            predicted_vjepa: [B, 2048, 1408]
        """
        batch_size = noisy_latents.shape[0]
        
        # 1. Encode noisy latents
        latent_features = self.latent_encoder(noisy_latents)  # [B, hidden_dim, 8, 16, 16]
        
        # Flatten spatial-temporal: [B, hidden_dim, 8, 16, 16] -> [B, 8*16*16, hidden_dim]
        latent_features = latent_features.flatten(2).transpose(1, 2)
        
        # 2. Project text embeddings
        text_features = self.text_proj(text_embeddings)  # [B, text_seq_len, hidden_dim]
        
        # 3. Get timestep embeddings
        time_emb = self.get_timestep_embedding(timesteps, self.hidden_dim)  # [B, hidden_dim]
        time_emb = self.time_proj(time_emb).unsqueeze(1)  # [B, 1, hidden_dim]
        
        # 4. Fuse: latent + text + time
        # Concatenate all features
        fused_features = torch.cat([latent_features, text_features, time_emb], dim=1)
        # Shape: [B, (latent_tokens + text_tokens + 1), hidden_dim]
        
        # Apply fusion transformer layers
        for layer in self.fusion_layers:
            fused_features = layer(fused_features)
        
        # 5. Decode to VJEPA tokens
        # Use learnable queries
        vjepa_queries = self.vjepa_queries.expand(batch_size, -1, -1)  # [B, 2048, hidden_dim]
        
        # Cross-attend to fused features
        vjepa_features = vjepa_queries
        for layer in self.vjepa_decoder:
            vjepa_features = layer(vjepa_features, fused_features)
        
        # 6. Project to VJEPA dimension
        predicted_vjepa = self.output_proj(vjepa_features)  # [B, 2048, 1408]
        
        return predicted_vjepa


class LatteTransformer3DModelWithPhysics(ModelMixin, ConfigMixin, CacheMixin):
    """
    Modified Latte with PredictorP for physics conditioning.
    
    Training mode: PredictorP predicts VJEPA, supervised by ground-truth VJEPA
    Inference mode: PredictorP predicts VJEPA (no ground-truth needed)
    """
    
    _supports_gradient_checkpointing = True
    _skip_layerwise_casting_patterns = ["pos_embed", "norm"]

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        cross_attention_dim: Optional[int] = None,
        attention_bias: bool = False,
        sample_size: int = 64,
        patch_size: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        norm_type: str = "layer_norm",
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        caption_channels: int = None,
        video_length: int = 16,
        # PredictorP parameters
        predictor_hidden_dim: int = 512,
        vjepa_dim: int = 1408,
        vjepa_seq_len: int = 2048,  # 16 frames: 256 spatial * 8 temporal patches
        use_predictor: bool = True,
    ):
        super().__init__()
        
        inner_dim = num_attention_heads * attention_head_dim
        self.use_predictor = use_predictor

        # 1. PredictorP network
        if use_predictor:
            self.predictor = PredictorP(
                latent_channels=in_channels,
                latent_frames=video_length,
                latent_height=sample_size,
                latent_width=sample_size,
                text_dim=caption_channels,
                hidden_dim=predictor_hidden_dim,
                vjepa_seq_len=vjepa_seq_len,
                vjepa_dim=vjepa_dim,
                num_layers=4,
            )
        else:
            self.predictor = None

        # 2. Define input layers
        self.height = sample_size
        self.width = sample_size
        interpolation_scale = self.config.sample_size // 64
        interpolation_scale = max(interpolation_scale, 1)
        
        self.pos_embed = PatchEmbed(
            height=sample_size,
            width=sample_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=inner_dim,
            interpolation_scale=interpolation_scale,
        )

        # 3. Define spatial transformers blocks (text conditioning)
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    cross_attention_dim=cross_attention_dim,
                    activation_fn=activation_fn,
                    num_embeds_ada_norm=num_embeds_ada_norm,
                    attention_bias=attention_bias,
                    norm_type=norm_type,
                    norm_elementwise_affine=norm_elementwise_affine,
                    norm_eps=norm_eps,
                )
                for d in range(num_layers)
            ]
        )

        # 4. Define temporal transformers blocks (VJEPA physics conditioning)
        self.temporal_transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    cross_attention_dim=inner_dim if use_predictor else None,  # VJEPA cross-attention
                    activation_fn=activation_fn,
                    num_embeds_ada_norm=num_embeds_ada_norm,
                    attention_bias=attention_bias,
                    norm_type=norm_type,
                    norm_elementwise_affine=norm_elementwise_affine,
                    norm_eps=norm_eps,
                )
                for d in range(num_layers)
            ]
        )

        # 5. VJEPA projection (project to inner_dim for temporal blocks)
        if use_predictor:
            self.vjepa_projection = nn.Sequential(
                nn.Linear(vjepa_dim, inner_dim * 2),
                nn.GELU(),
                nn.Linear(inner_dim * 2, inner_dim),
            )
        else:
            self.vjepa_projection = None

        # 6. Output layers
        self.out_channels = in_channels if out_channels is None else out_channels
        self.norm_out = nn.LayerNorm(inner_dim, elementwise_affine=False, eps=1e-6)
        self.scale_shift_table = nn.Parameter(torch.randn(2, inner_dim) / inner_dim**0.5)
        self.proj_out = nn.Linear(inner_dim, patch_size * patch_size * self.out_channels)

        # 7. Other Latte blocks
        self.adaln_single = AdaLayerNormSingle(inner_dim, use_additional_conditions=False)
        self.caption_projection = PixArtAlphaTextProjection(in_features=caption_channels, hidden_size=inner_dim)

        # 8. Temporal positional embedding
        temp_pos_embed = get_1d_sincos_pos_embed_from_grid(
            inner_dim, torch.arange(0, video_length).unsqueeze(1), output_type="pt"
        )
        self.register_buffer("temp_pos_embed", temp_pos_embed.to(torch.bfloat16).unsqueeze(0), persistent=False)
        
        self.gradient_checkpointing = True

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: Optional[torch.LongTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        ground_truth_vjepa: Optional[torch.Tensor] = None,  # For training
        enable_temporal_attentions: bool = True,
        return_dict: bool = True,
    ):
        """
        Forward pass with PredictorP.
        
        Args:
            hidden_states: Noisy latents [B, T, C, H, W] = [B, 16, 4, 32, 32]
            timestep: Diffusion timestep
            encoder_hidden_states: Text embeddings [B, seq_len, text_dim]
            ground_truth_vjepa: Ground-truth VJEPA tokens [B, 2048, 1408] (for training only)
            
        Returns:
            Transformer2DModelOutput with predicted_vjepa if use_predictor=True
        """
        batch_size, channels, num_frame, height, width = hidden_states.shape
        
        # ========== STEP 1: Predict VJEPA tokens using PredictorP ==========
        predicted_vjepa = None
        if self.use_predictor and self.predictor is not None:
            predicted_vjepa = self.predictor(
                noisy_latents=hidden_states,
                text_embeddings=encoder_hidden_states,
                timesteps=timestep,
            )  # [B, 2048, 1408]
            
            # Use ground-truth during training if provided, else use predicted
            if ground_truth_vjepa is not None and self.training:
                physics_tokens = ground_truth_vjepa  # Teacher forcing
            else:
                physics_tokens = predicted_vjepa  # Use prediction
        else:
            physics_tokens = ground_truth_vjepa if ground_truth_vjepa is not None else None
        
        # ========== STEP 2: Standard Latte processing ==========
        # Reshape: (B, C, T, H, W) -> (B*T, C, H, W)
        hidden_states = hidden_states.permute(0, 2, 1, 3, 4).reshape(-1, channels, height, width)

        # Patch embedding
        height, width = (
            hidden_states.shape[-2] // self.config.patch_size,
            hidden_states.shape[-1] // self.config.patch_size,
        )
        num_patches = height * width
        hidden_states = self.pos_embed(hidden_states)

        # Timestep embedding
        added_cond_kwargs = {"resolution": None, "aspect_ratio": None}
        timestep, embedded_timestep = self.adaln_single(
            timestep, added_cond_kwargs=added_cond_kwargs, batch_size=batch_size, hidden_dtype=hidden_states.dtype
        )

        # Prepare text for spatial blocks
        encoder_hidden_states = self.caption_projection(encoder_hidden_states)
        encoder_hidden_states_spatial = encoder_hidden_states.repeat_interleave(
            num_frame, dim=0
        ).view(-1, encoder_hidden_states.shape[-2], encoder_hidden_states.shape[-1])

        # Prepare VJEPA for temporal blocks
        physics_embeddings_temporal = None
        if physics_tokens is not None and self.vjepa_projection is not None:
            physics_projected = self.vjepa_projection(physics_tokens)  # [B, 2048, inner_dim]
            # physics_embeddings_temporal = physics_projected.repeat_interleave(
            #     num_patches, dim=0
            # ).view(-1, physics_projected.shape[-2], physics_projected.shape[-1])
            physics_embeddings_temporal = physics_projected

        # Prepare timesteps
        timestep_spatial = timestep.repeat_interleave(num_frame, dim=0).view(-1, timestep.shape[-1])
        timestep_temp = timestep.repeat_interleave(num_patches, dim=0).view(-1, timestep.shape[-1])

        # ========== STEP 3: Spatial + Temporal blocks ==========
        for i, (spatial_block, temp_block) in enumerate(
            zip(self.transformer_blocks, self.temporal_transformer_blocks)
        ):
            # Spatial block (text conditioning)
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(
                    spatial_block,
                    hidden_states,
                    None,  # attention_mask
                    encoder_hidden_states_spatial,
                    encoder_attention_mask,
                    timestep_spatial,
                    None,  # cross_attention_kwargs
                    None,  # class_labels
                )
            else:
                hidden_states = spatial_block(
                    hidden_states,
                    None,
                    encoder_hidden_states_spatial,
                    encoder_attention_mask,
                    timestep_spatial,
                    None,
                    None,
                )

            if enable_temporal_attentions:
                # Reshape for temporal: (B*T, H*W, C) -> (B*H*W, T, C)
                hidden_states = hidden_states.reshape(
                    batch_size, -1, hidden_states.shape[-2], hidden_states.shape[-1]
                ).permute(0, 2, 1, 3)
                hidden_states = hidden_states.reshape(-1, hidden_states.shape[-2], hidden_states.shape[-1])

                # Add temporal pos embedding
                if i == 0 and num_frame > 1:
                    hidden_states = hidden_states + self.temp_pos_embed.to(hidden_states.dtype)

                # Temporal block (VJEPA physics conditioning!)
                if torch.is_grad_enabled() and self.gradient_checkpointing:
                    hidden_states = self._gradient_checkpointing_func(
                        temp_block,
                        hidden_states,
                        None,  # attention_mask
                        physics_embeddings_temporal,  # ← VJEPA physics!
                        None,  # encoder_attention_mask
                        timestep_temp,
                        None,  # cross_attention_kwargs
                        None,  # class_labels
                    )
                else:
                    hidden_states = temp_block(
                        hidden_states,
                        None,
                        physics_embeddings_temporal,  # ← VJEPA physics!
                        None,
                        timestep_temp,
                        None,
                        None,
                    )

                # Reshape back: (B*H*W, T, C) -> (B*T, H*W, C)
                hidden_states = hidden_states.reshape(
                    batch_size, -1, hidden_states.shape[-2], hidden_states.shape[-1]
                ).permute(0, 2, 1, 3)
                hidden_states = hidden_states.reshape(-1, hidden_states.shape[-2], hidden_states.shape[-1])

        # ========== STEP 4: Output ==========
        embedded_timestep = embedded_timestep.repeat_interleave(num_frame, dim=0).view(-1, embedded_timestep.shape[-1])
        shift, scale = (self.scale_shift_table[None] + embedded_timestep[:, None]).chunk(2, dim=1)
        hidden_states = self.norm_out(hidden_states)
        hidden_states = hidden_states * (1 + scale) + shift
        hidden_states = self.proj_out(hidden_states)

        # Unpatchify
        hidden_states = hidden_states.reshape(
            shape=(-1, height, width, self.config.patch_size, self.config.patch_size, self.out_channels)
        )
        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
        output = hidden_states.reshape(
            shape=(-1, self.out_channels, height * self.config.patch_size, width * self.config.patch_size)
        )
        
        # Reshape to video
        output = output.reshape(batch_size, -1, output.shape[-3], output.shape[-2], output.shape[-1]).permute(
            0, 2, 1, 3, 4
        )

        if not return_dict:
            return (output, predicted_vjepa) if predicted_vjepa is not None else (output,)
        
        return Transformer2DModelOutput(sample=output), predicted_vjepa