# PhysVideoGenerator — Continuation Handoff v2

**Read `HANDOFF.md` first** for the root-cause audit, why each bug matters, and the original 10-phase plan. This doc is the incremental update: what's been landed as of this session, what remains, and any deviations from HANDOFF v1.

**Status:** Code-only fixes from HANDOFF Phases 1, 2, 3, 5, 6, 7 are landed and verified locally. No commits yet. No new encoding, training, or evaluation has been run — the compute-heavy phases (4 re-encode, 8 smoke tests, 9 training, 10 eval) are still ahead.

**Branch:** `fix/pipeline_changes_and_evaluation`. HEAD at session start was `5c8d0bd docs: add HANDOFF.md`. Working tree carries 5 uncommitted edited files (see §3).

---

## 1. What was landed in this session

All CPU-only edits from the HANDOFF plan. GPU-required tests deferred.

| HANDOFF phase | Change | Files |
|---|---|---|
| 1 | Model config defaults `out_channels=8`, `attention_bias=True` at signature and every instantiation site | `src/latte_physics.py:248,252`; `src/train_latte_physics.py:154,158`; `evaluation/generate_physics.py:91,95`; `evaluation/generate_physics_v2.py:67,71` |
| 1 | Preflight assertion: `RuntimeError` if any pretrained key silently drops during shape-matched load | `src/train_latte_physics.py:192-204` |
| 1 (corollary) | Removed `_pad_sigma` from `PhysicsPipelineAdapter`. Model now outputs 8 channels natively so `LattePipeline`'s `chunk(2)[0]` correctly extracts noise. HANDOFF didn't call this out — was silently required by the Phase 1 output-channel change | `evaluation/generate_physics_v2.py` (former lines 150-160) |
| 2 | `DDPMScheduler.from_pretrained("maxin-cn/Latte-1", subfolder="scheduler")` replaces hand-rolled scaled_linear schedule | `src/train_latte_physics.py:275` |
| 3 | Removed manual ImageNet normalization from VJEPA branch. `AutoVideoProcessor` inside `VJEPA2Encoder` handles it downstream | `src/datasets/clean_videos.py:161-167` |
| 5 | Loss slices `model_output.sample[:, :4]` (noise half) and casts both operands to `.float()` | `src/train_latte_physics.py:428-434` |
| 6 | 10% classifier-free-guidance null-prompt dropout before model forward | `src/train_latte_physics.py:409-413` |
| 7 | `pipe.tokenizer.model_max_length = 226` to match training-time T5 sequence length | `evaluation/generate_physics_v2.py:167-168` |

**Verified:** `grep` confirms every out_channels/attention_bias value is correct, `_pad_sigma` is removed, all 5 files AST-parse clean.

---

## 2. What remains

Ordered by dependency. Each row is one session of work.

### Session A — VJEPA temporal-layout probe (GPU, ~15 min)

Verify VJEPA's 2048 tokens are laid out `[B, 8_temporal, 256_spatial, D]` (what `src/latte_physics.py:465-466` assumes) or the transposed layout. Independent of the re-encoded dataset — probe uses synthetic frames straight through the encoder. Script is in HANDOFF §4 Phase 3, second code block.

**If probe result is spatial-outer:** flip the reshape order in `src/latte_physics.py:465-466` before Phase 4 or Phase 9.

This is the single riskiest step. Getting it wrong wastes the 30-hour training run.

### Session B — Re-encode dataset (GPU, 5-10 hr)

```bash
rm -rf data/encoded_videos/*
python src/iterative_encode_curated.py --hf_repo Boxxxi/physvideogen-encoded-v2
```

Both `--hf_repo` (encoder) and `--hf_repo` (trainer) are already wired via argparse — no code change needed.

**Sanity check after:** VJEPA npz mean/std should visibly differ from the old `Boxxxi/physvideogen-encoded` VJEPA tensors — that's the double-normalization fix biting. If they match, the fix didn't take effect.

### Session C — Smoke tests (GPU, 1-2 hr)

HANDOFF §4 Phase 8 tests 1-4:
1. Preflight assertion passes for pretrained load. (This is the guardrail added in Session-past Phase 1.)
2. Untrained physics model (fresh init, `use_predictor=False`) matches `generate_baseline.py` output for same seed/prompt.
3. 100-sample 1-epoch training: no NaN, `noise_loss` trends down, output shape `[B, 8, T, H, W]`.
4. 1-epoch checkpoint through `generate_physics_v2.py` produces baseline-adjacent output (not pure noise).

Do not proceed to Session D until all four pass.

### Session D — Full training (GPU, ~30 hr, chunked)

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

Chunk across Colab sessions with `--resume_from_checkpoint <dir>`. Watch loss curves per HANDOFF §4 Phase 9.

### Session E — Evaluation (GPU, 2-3 hr)

```bash
python evaluation/generate_physics_v2.py --checkpoint checkpoints_run2/checkpoint_epoch_20.pt
```

Then manual visual comparison against baseline videos, and optional VideoPhy PC/SA scores.

---

## 3. Files changed (uncommitted)

```
src/latte_physics.py                   # Phase 1 signature defaults
src/train_latte_physics.py             # Phases 1, 2, 5, 6 (instantiation, preflight, scheduler, loss, CFG)
src/datasets/clean_videos.py           # Phase 3 double-norm removal
evaluation/generate_physics.py         # Phase 1 instantiation
evaluation/generate_physics_v2.py      # Phase 1 instantiation + _pad_sigma removal + Phase 7 tokenizer length
```

Suggested commit split (if you want atomic history):
- `fix: correct Latte-1 model config (out_channels, attention_bias) and add preflight assertion`
- `fix: use Latte-1's shipped noise scheduler for training`
- `fix: remove VJEPA double-normalization from data pipeline`
- `fix: slice noise loss + fp32 cast + CFG null-prompt dropout`
- `fix: force T5 max_length=226 and unpad model output in v2 inference`

Or one bundled `fix: apply HANDOFF phases 1-3, 5-7 code edits`.

---

## 4. Deviations from HANDOFF v1

1. **HANDOFF §4 Phase 1 line references drifted** — the model instantiation in `train_latte_physics.py` is at line 150 (not 116-139); the shape filter is at line 189 (not 154-156); the scheduler block is at line 259 (not 236-242); the loss block is at line 411 (not 370-373); text embeddings used at line 401 (not 361-367). All were located by pattern, not line number.

2. **HANDOFF §3 identified `src/datasets/clean_videos.py:180-184` for the VJEPA double-norm fix** — actual location was 163-166 in this session's tree. Behavior of the edit matches.

3. **HANDOFF Phase 1 didn't mention the `_pad_sigma` adapter change** in `evaluation/generate_physics_v2.py`. Once `out_channels=8` lands in the model, the adapter's pre-existing padding logic would double the channel count to 16 and break v2 inference. Removing it is not optional — it's part of the same change. This session made the removal alongside the config fix.

4. **HANDOFF Phase 1 signature-default change** — `latte_physics.py` had `out_channels: Optional[int] = None` (not `= 4` as HANDOFF described). Callers pass explicit values, so the default was functionally unreached. Set to `= 8` per HANDOFF's belt-and-suspenders intent.

5. **VJEPA layout probe (HANDOFF §4 Phase 3, second code block) was NOT run** in this session — it needs GPU. Deferred to Session A above. This is a real dependency for Phase 9 training success.

6. **Dead attributes on `VideoProcessor`** — `self.vjepa_mean` and `self.vjepa_std` are unused after the Phase 3 edit. Left in place to keep the diff narrow. Safe to delete when convenient.

---

## 5. HF resource state (as of session end)

- `Boxxxi/physvideogen-encoded` — old, double-normalized VJEPA. Keep as rollback. Do not delete.
- `Boxxxi/physvideogen-encoded-v2` — **not yet created.** Will be created by Session B encoding.
- `Boxxxi/physvideogen-checkpoints` — old run-1 weights. Do not reuse. Keep as rollback.
- `Boxxxi/physvideogen-checkpoints-v2` — **not yet created.** Will be created by Session D training.

`checkpoints_run1/` is a local artifact of the failed first training pass. Do not resume from it; do not delete it in case anyone wants to inspect the trained-on-corrupted-backbone weights.

---

## 6. Success criteria (unchanged from HANDOFF §5)

Any of the three counts as project success:

1. Physics videos are coherent AND show improved physical realism vs baseline (measured by VideoPhy PC or manual annotation) — positive result.
2. Physics videos are coherent but comparable/worse than baseline — well-characterized negative result.
3. PC improves but SA drops — real physics/text-adherence trade-off finding.

Failure = physics videos incoherent (like run 1). Would indicate further bugs beyond the audit.
