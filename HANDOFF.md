# PhysVideoGenerator — Continuation Handoff

**Status:** First training run (5 epochs, `checkpoints_run1/`) produced degenerate inference output. Root causes identified via architectural audit. This document is the complete plan to fix and successfully train the physics-conditioned model.

**Goal of the project:** Condition Latte-1 (video diffusion transformer) on VJEPA-2 (self-supervised video representations) to inject physical-motion priors into text-to-video generation. Physics head trains, backbone stays frozen.

**Compute assumption:** Unbounded — plan requires ~45-50 GPU-hours total across ~5-7 days of chunked Colab A100 access, HF for storage.

---

## 1. Current repo state

- **Branch to work on:** `fix/pipeline_changes_and_evaluation`
- **Working baseline:** `evaluation/generate_baseline.py` — vanilla LattePipeline, produces 25 coherent videos from `evaluation/eval_prompts.csv`. Do not modify.
- **Broken physics inference:** `evaluation/generate_physics_v2.py` — LattePipeline + physics-transformer adapter. Runs, produces incoherent output. This is what needs to work after the fixes.
- **Broken training:** `src/train_latte_physics.py` — completes without errors but trains against a corrupted backbone (see §3).
- **Data pipeline:** `src/iterative_encode_curated.py` — encodes curated OpenVid-1M clips to VAE/VJEPA/text tensors. Works correctly for VAE + text; has a VJEPA double-normalization bug (see §3).
- **Model class:** `src/latte_physics.py` — `LatteTransformer3DModelWithPhysics` with `PredictorP` + `PhysicsFrameCrossAttn`. Architecture is sound; config has two silent-loading bugs.

**Existing checkpoints (do not reuse):** `Boxxxi/physvideogen-checkpoints` on HF has epochs 1-5 from the broken training. These weights are calibrated for a corrupted backbone and cannot be recovered. Retrain from scratch after fixes.

**Existing encoded data (do not reuse):** `Boxxxi/physvideogen-encoded` has VJEPA tokens computed from doubly-normalized inputs. Re-encode after fixing §3 (Fix 3).

---

## 2. What was previously done (session summary)

1. Reviewed inherited code, identified 5 architectural flaws (see git log for `fix:` commits on this branch).
2. Fixed Flaw 1 (inference tensor layout permute), Flaw 2 (scheduled sampling replacing hard teacher forcing), Flaw 4 (per-frame VJEPA cross-attention replacing 2048-token broadcast).
3. Migrated resolution from 256×256 → 512×512 (Latte-1's pretrained resolution). Split VideoProcessor into vae_frame_size=512 / vjepa_frame_size=256.
4. Built iterative encoding pipeline using `RemoteZip` for HTTP-range streaming of OpenVid-1M zips.
5. Trained 5 epochs with scheduled sampling curriculum.
6. Baseline generation works. Physics inference produces degenerate output regardless of `use_predictor=True/False`.
7. Independent audit identified the actual root causes below.

---

## 3. Root causes to fix

Ranked by severity. All must be fixed before retraining.

### Critical: `out_channels=4` in model config vs Latte-1's actual `out_channels=8`
- **Where:** `src/latte_physics.py:184-208` `__init__` signature default, and `src/train_latte_physics.py:120`, `evaluation/generate_baseline.py`, all inference scripts.
- **Consequence:** Pretrained `proj_out.weight` shape `[32, 1152]` (for 8 output channels) doesn't match our model's `[16, 1152]`. Shape-matching filter at `train_latte_physics.py:154-156` silently drops it. `proj_out` stays random-initialized. It's frozen (not in trainable set). Every forward pass ends with a random Linear projection.
- **Verification:** Latte-1 config on HF confirms `out_channels: 8` (predicts noise + learned sigma per DDPM).

### Critical: `attention_bias=False` in model config vs Latte-1's actual `attention_bias=True`
- **Where:** Same instantiation sites as above.
- **Consequence:** We don't allocate q/k/v/out bias params. Pretrained biases exist in the checkpoint but silently can't load. 28 layers × 2 attn blocks × 4 projections = ~224 missing bias tensors per model. Frozen spatial blocks run pretrained weights without their trained biases.

### High: Training DDPM schedule doesn't match Latte-1's shipped scheduler
- **Where:** `src/train_latte_physics.py:236-242` uses `beta_schedule="scaled_linear", beta_start=0.0001, beta_end=0.02`.
- **Consequence:** Latte-1's actual scheduler config is `beta_schedule="linear"`. Different alpha_bar curve. Training reverses one noise process; inference uses a different one.

### High: VJEPA input frames are doubly-normalized
- **Where:** `src/datasets/clean_videos.py:180-184` applies ImageNet normalization to the VJEPA branch. Then `src/encoders/vjepa2_encoder.py:31-34` passes those already-normalized frames through `AutoVideoProcessor`, which normalizes again.
- **Consequence:** Every VJEPA token was computed from out-of-distribution input to VJEPA-2. Physics signal is semantically null even though loss shape is fine.

### Medium: `LattePipeline` default T5 `max_length=120` vs training's 226
- **Where:** `pipeline_latte.py` internal default (not under our control) vs `src/encoders/text_caption_enocder.py:26` default and calls with `max_length=226`.
- **Consequence:** Training conditioned on 226-token embeddings; v2 inference feeds 120-token embeddings through `caption_projection`. Not fatal but a real train/inference mismatch.

### Medium: No CFG null-prompt during training
- **Where:** `src/train_latte_physics.py:361-367` always passes real text embeddings, never drops for null-prompt training.
- **Consequence:** At inference with CFG > 1, PredictorP has never seen null-prompt inputs; predicts garbage for the uncond branch which gets amplified by guidance scale.

### Minor: MSE loss cast to bf16 truncates gradient signal
- **Where:** `src/train_latte_physics.py:370-373` `.to(torch.bfloat16)` before `F.mse_loss`.
- **Consequence:** Loss computation in bf16's 7-bit mantissa loses small gradient magnitudes.

---

## 4. Execution plan

Execute phases in order. Do not skip Phase 8 (smoke tests) — it's the guardrail against wasting 30 hours of training on unfixed bugs.

### Phase 1 — Model config fixes (30 min coding)

**Edit `src/latte_physics.py`:**
```python
# In LatteTransformer3DModelWithPhysics.__init__ signature, change defaults:
out_channels: int = 8,       # was 4
attention_bias: bool = True, # was False
```

**Edit every model instantiation** to match the new defaults or pass explicit `out_channels=8, attention_bias=True`. Files to update:
- `src/train_latte_physics.py:116-139`
- `evaluation/generate_baseline.py` — does NOT need changes (uses LattePipeline, gets vanilla Latte-1 config)
- `evaluation/generate_physics.py:82-107` (build_physics_model function)
- `evaluation/generate_physics_v2.py:82-107` (build_physics_model function)

**Add a preflight assertion in `src/train_latte_physics.py` immediately after the pretrained loading loop:**
```python
# Preflight: verify pretrained keys all load. This assertion would have caught
# the original out_channels=4 and attention_bias=False bugs.
loaded_keys = 0
for key in pretrained_state:
    if key in model_state and model_state[key].shape == pretrained_state[key].shape:
        loaded_keys += 1
missing = [k for k in pretrained_state if k not in model_state or model_state[k].shape != pretrained_state[k].shape]
if missing:
    print(f"WARNING: {len(missing)} pretrained keys silently dropped. First 10:")
    for k in missing[:10]:
        model_shape = model_state[k].shape if k in model_state else "MISSING"
        print(f"  {k}: pretrained={pretrained_state[k].shape} model={model_shape}")
    raise RuntimeError("Pretrained loading incomplete. Fix config and retry.")
print(f"Pretrained loading: {loaded_keys}/{len(pretrained_state)} keys loaded cleanly.")
```

### Phase 2 — Noise schedule fix (5 min)

**Edit `src/train_latte_physics.py`** around line 236, replace the hard-coded DDPM instantiation:
```python
from diffusers import DDPMScheduler
noise_scheduler = DDPMScheduler.from_pretrained("maxin-cn/Latte-1", subfolder="scheduler")
```
This matches Latte-1's actual training schedule.

### Phase 3 — VJEPA preprocessing fix (15 min coding)

**Edit `src/datasets/clean_videos.py`** in `process_video`, remove the manual ImageNet normalization from the VJEPA branch:
```python
# Pipeline B: VJEPA — just resize + [0,1] float; AutoVideoProcessor normalizes downstream
vjepa_video = F.resize(cropped, [self.vjepa_frame_size, self.vjepa_frame_size])
vjepa_video = vjepa_video / 255.0  # to [0, 1] float
# NO manual per-channel normalization — AutoVideoProcessor does this inside the encoder
```

Also **verify VJEPA-2's output token layout** before re-encoding. Our reshape at `src/latte_physics.py:465-466` assumes `[B, 8_temporal, 256_spatial, D]`. Verify with this test script:
```python
# Encode two synthetic 16-frame "videos" and check token layout
import torch
from src.encoders.vjepa2_encoder import VJEPA2Encoder
enc = VJEPA2Encoder("facebook/vjepa2-vitg-fpc64-256", torch.float16, "cuda")

# Video A: first 8 frames uniform gray, last 8 frames uniform white
frames = torch.zeros(16, 3, 256, 256)
frames[:8] = 0.5    # gray for first half
frames[8:] = 1.0    # white for second half

with torch.inference_mode():
    tokens = enc.encode(frames.to("cuda"))  # [1, 2048, 1408]

# Reshape assuming temporal-outer: [B, 8_temporal, 256_spatial, D]
temporal_outer = tokens.reshape(1, 8, 256, 1408)
# temporal_outer[:, 0, :, :] should be all-similar (all from gray frames)
# temporal_outer[:, 7, :, :] should be all-similar (all from white frames)
# and they should differ between t=0 and t=7
similarity_within_t0 = torch.corrcoef(temporal_outer[0, 0, :50, :].flatten(1))[0, 1:].mean()
similarity_across = (temporal_outer[0, 0, :50, :] - temporal_outer[0, 7, :50, :]).abs().mean()
print(f"Within t=0: {similarity_within_t0:.3f}  Across t=0 vs t=7: {similarity_across:.3f}")

# If temporal-outer: within is high, across is high (differ). If spatial-outer: opposite pattern.
# Update src/latte_physics.py reshape accordingly.
```

### Phase 4 — Re-encode dataset (5-10 hours compute)

Push existing encoded data to a `-v1` HF repo as backup, then re-encode fresh:
```bash
# Backup existing (already on HF as Boxxxi/physvideogen-encoded)
# Delete local
rm -rf data/encoded_videos/*

# Re-encode with fixed VJEPA preprocessing into a NEW HF repo
python src/iterative_encode_curated.py --hf_repo Boxxxi/physvideogen-encoded-v2
```

The `-v2` suffix means the old data is preserved on HF as a rollback if something goes wrong. Verify the fixed encoding worked by checking a random VJEPA sample has different token statistics than the old encoding:
```bash
python -c "
import numpy as np, glob
old = np.load(sorted(glob.glob('...old npz path...'))[0])['arr_0']
new = np.load(sorted(glob.glob('data/encoded_videos/vjepa/*.npz'))[0])['arr_0']
print(f'Old: mean={old.mean():.3f} std={old.std():.3f}')
print(f'New: mean={new.mean():.3f} std={new.std():.3f}')
# Distributions should differ if double-normalization was actually happening
"
```

### Phase 5 — Fix training loss (10 min coding)

**Edit `src/train_latte_physics.py`** in the training loop, where noise_loss is computed:
```python
# Model now outputs 8 channels (noise + learned variance).
# Slice to first 4 (noise channels) for the diffusion loss.
# Also compute in fp32 for MSE numerical stability.
model_output_noise = model_output.sample[:, :4]  # noise channels
noise_loss = F.mse_loss(model_output_noise.float(), noise.float())
vjepa_loss = F.mse_loss(predicted_vjepa.float(), vjepa_gt.float())
total_loss = noise_loss + vjepa_loss_weight * vjepa_loss
```
The variance channels (4-7) are ignored — training only supervises noise prediction. `LattePipeline` extracts noise via `chunk(2, dim=1)[0]` = first 4 channels, so this alignment is correct.

### Phase 6 — Add CFG null-prompt training (10 min coding)

**Edit `src/train_latte_physics.py`** in the training loop, before the model forward:
```python
# 10% chance to drop text conditioning for classifier-free-guidance training.
# Without this, PredictorP and physics_cross_attn never see the null-prompt
# distribution and at inference (CFG > 1) they produce garbage for the uncond branch.
if torch.rand((), device=text_embeddings.device).item() < 0.1:
    text_embeddings = torch.zeros_like(text_embeddings)
```

### Phase 7 — Fix inference-time text length (5 min coding)

**Edit `evaluation/generate_physics_v2.py`** after loading the pipeline:
```python
# Force LattePipeline to use same T5 max_length as training (226 tokens)
pipe.tokenizer.model_max_length = 226
```

Verify the setting took effect by printing `pipe.tokenizer.model_max_length` before generation.

### Phase 8 — Smoke tests (1-2 hours)

**Do not skip.** Each test either passes or produces a specific failure that must be fixed before proceeding.

**Test 1: Pretrained loading is clean.** Run the assertion added in Phase 1. Expected: all pretrained keys load. Failure means Phase 1 fix is incomplete.

**Test 2: Untrained physics model reproduces baseline behavior.**
```bash
# Instantiate the fixed model (out_channels=8, attention_bias=True, pretrained-loaded).
# Save a "fresh init" checkpoint before any training.
# Then run inference through generate_physics_v2 with this fresh checkpoint AND use_predictor=False.
# Output should be visually indistinguishable from evaluation/generate_baseline.py output for the same prompt/seed.
```
Failure indicates the physics model's forward path is still doing something the vanilla model doesn't. Iterate.

**Test 3: 100-sample training completes without NaN.**
```bash
python src/train_latte_physics.py --num_samples 100 --epochs 1 --output_dir smoke_test_v2 --lr 1e-5 --vjepa_weight 0.3
```
Watch losses. noise_loss should trend down. Neither loss should NaN. Model output shapes should match expected `[B, 8, T, H, W]`.

**Test 4: Fresh 1-epoch checkpoint through inference produces coherent output.**
Load the 1-epoch checkpoint into generate_physics_v2, run on prompt_id 1. Should not be pure noise. Should be baseline-adjacent (physics head has trained 1 epoch, mostly teacher-forced, minimal deviation).

If all four pass, proceed to Phase 9. If any fail, do not proceed — root-cause the failure first.

### Phase 9 — Full training run (~30 hours compute)

Fresh training from scratch. Do not resume from `checkpoints_run1/` — those weights are calibrated for the corrupted backbone.

```bash
python src/train_latte_physics.py \
    --index_json data/indexed_dataset.json \
    --epochs 20 \
    --gradient_accumulation_steps 1 \
    --lr 1e-5 \
    --vjepa_weight 0.3 \
    --tf_warmup_frac 0.15 \
    --tf_anneal_frac 0.5 \
    --output_dir ./checkpoints_run2 \
    --hf_repo Boxxxi/physvideogen-checkpoints-v2
```

Key hyperparameter changes from run 1:
- `--epochs 20` — gives 5 full epochs of pure closed-loop training (last 25%).
- `--gradient_accumulation_steps 1` — memory should fit at fp16 now.
- `--vjepa_weight 0.3` — aux loss was dominating at 1.0.
- `--tf_warmup_frac 0.15` — shorter warmup, more anneal + closed-loop.
- Fresh output dir and HF repo (`-v2` suffix) — no accidental resume from broken checkpoints.

Watch during training:
- Epoch 1: noise_loss should drop faster than run 1 (backbone is now correctly loaded).
- Epochs 2-14: noise_loss may rise during anneal — expected, model is being challenged with predicted VJEPA.
- Epochs 15-20 (closed-loop): noise_loss should stabilize, not explode. vjepa_loss should approach a floor.

Save every epoch. Auto-push to HF via `--hf_repo` handles cross-session recovery.

### Phase 10 — Evaluation (2-3 hours)

**Generate physics videos for all 25 prompts:**
```bash
python evaluation/generate_physics_v2.py \
    --checkpoint checkpoints_run2/checkpoint_epoch_20.pt
```

**Manual visual comparison** — for each prompt id, watch both `evaluation/baseline_videos/<id>_*.mp4` and `evaluation/physics_videos/<id>_*.mp4`. Same seed = matched initial noise. Any perceptual difference is attributable to the physics head.

**VideoPhy2 objective metrics** — optional but strong for a research writeup:
```bash
cd src/evaluate/VideoPhy/VIDEOPHY2
pip install -r requirements.txt
python inference.py --video_dir ../../../../evaluation/baseline_videos ...
python inference.py --video_dir ../../../../evaluation/physics_videos ...
```
Compare mean Physical Commonsense (PC) and Semantic Adherence (SA) scores.

---

## 5. Success criteria

Any of the following counts as a successful project outcome:

1. **Physics videos are coherent AND show improved physical realism** on physics prompts vs baseline (measured by VideoPhy PC or manual annotation). Publishable positive result.
2. **Physics videos are coherent but comparable/worse** to baseline on physics prompts. Suggests physics conditioning at this budget doesn't help; well-characterized negative result.
3. **PC improves but SA drops** — physics conditioning trades text-adherence for physics-realism. Real trade-off finding.

Failure = physics videos are incoherent (like run 1). This indicates further bugs beyond the audit.

---

## 6. Environment / compute setup

- **Compute:** Colab A100 (Pro or Pro+). ~30 GPU-hours for training, 5-10 for encoding, 3-5 for evaluation.
- **Session persistence:** Use HF for storage. `Boxxxi/physvideogen-encoded-v2` for data, `Boxxxi/physvideogen-checkpoints-v2` for checkpoints. Both survive session death; re-download on new session start.
- **HF auth:** Use Colab userdata pattern (`os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN')` before any HF operation). More reliable than `hf auth login`.
- **Dependencies:** `pip install -q remotezip hf_transfer huggingface_hub diffusers[torch] accelerate transformers`.
- **If Xet CAS errors return:** `pip install -U huggingface_hub hf-xet` fixed it last time.

---

## 7. Known pitfalls (from prior session)

1. **HF Xet CAS returns 403 with "invalid key pair id"** — HF-side edge config issue. Try: fresh runtime, updated `huggingface_hub + hf-xet`, use userdata token pattern.
2. **Colab session drops mid-training** — `--resume_from_checkpoint <dir>` works but re-download data first.
3. **VJEPA temporal layout assumption** — Verify BEFORE re-encoding (Phase 3). Getting this wrong means every physics frame gathers wrong tokens.
4. **Silent shape-dropping in pretrained loading** — The assertion added in Phase 1 catches this. Without it, this exact bug class caused the entire failed run 1.
5. **`generate_baseline.py` no skip-if-exists** — On partial reruns, either delete existing videos or add skip logic.
6. **fp16 vs bf16 mixed usage** — Training uses bf16 (accelerator), generate_physics_v2 uses fp16 (matching pipeline). Model dtype casts happen at load time. If dtype errors surface, check where the tensor originated.

---

## 8. File reference

**Model:**
- `src/latte_physics.py` — `LatteTransformer3DModelWithPhysics` (contains PredictorP + PhysicsFrameCrossAttn)

**Training:**
- `src/train_latte_physics.py` — training loop with scheduled sampling, HF checkpoint push
- `src/datasets/clean_videos.py` — `VideoProcessor` (video → VAE + VJEPA branches)

**Encoding:**
- `src/iterative_encode_curated.py` — full data prep pipeline (RemoteZip streaming, HF push)
- `src/encoders/vae_encoder_decoder.py` — VAE wrapper
- `src/encoders/vjepa2_encoder.py` — VJEPA-2 wrapper
- `src/encoders/text_caption_enocder.py` — T5-XXL wrapper

**Inference:**
- `evaluation/generate_baseline.py` — vanilla Latte-1 baseline (works, do not modify)
- `evaluation/generate_physics.py` — v1 custom denoising loop (has plumbing issues, deprecated)
- `evaluation/generate_physics_v2.py` — v2 LattePipeline + adapter (this is the one to fix)
- `evaluation/eval_prompts.csv` — 25 evaluation prompts with categories

**Data:**
- `data/text_csv/curated_OpenVid-1M.csv` — 4529 curated clips with captions
- `data/text_csv/OpenVid-1M.csv` — full OpenVid-1M metadata
- `data/encoded_videos/{vae,vjepa,text}/` — encoded tensors
- `data/indexed_dataset.json` — training index (built by `iterative_encode_curated.py`)

**External:**
- `src/evaluate/VideoPhy/VIDEOPHY2/` — VideoPhy v2 for objective evaluation

**HF resources:**
- `Boxxxi/physvideogen-encoded` (old, keep as backup)
- `Boxxxi/physvideogen-encoded-v2` (new, create in Phase 4)
- `Boxxxi/physvideogen-checkpoints` (old, do not reuse)
- `Boxxxi/physvideogen-checkpoints-v2` (new, create in Phase 9)

---

## 9. Do NOT do

- Do not modify `evaluation/generate_baseline.py`. It works. Do not touch.
- Do not resume training from `checkpoints_run1/` (or the old HF checkpoints repo). Those weights are calibrated for the corrupted backbone.
- Do not skip Phase 8 smoke tests. 30 hours of wasted training is worse than 2 hours of smoke tests.
- Do not overwrite `Boxxxi/physvideogen-encoded` or `Boxxxi/physvideogen-checkpoints` — the `-v2` suffix on new repos is intentional for rollback.
- Do not try to hand-patch inference to make the old checkpoint work. It fundamentally can't; the physics head trained against a corrupted backbone.
