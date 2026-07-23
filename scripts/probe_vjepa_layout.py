"""VJEPA-2 token layout probe. See HANDOFF §4 Phase 3."""
import torch
from src.encoders.vjepa2_encoder import VJEPA2Encoder

enc = VJEPA2Encoder("facebook/vjepa2-vitg-fpc64-256", torch.float16, "cuda")

# 16 frames: first 8 gray, last 8 white. Uniform in space, sharp step in time.
frames = torch.zeros(16, 3, 256, 256)
frames[:8] = 0.5
frames[8:] = 1.0

with torch.inference_mode():
    tokens = enc.encode(frames.to("cuda"))
print(f"tokens.shape = {tuple(tokens.shape)}")   # expect [1, 2048, 1408]

t = tokens[0].float()  # [2048, D]

# Hypothesis A (temporal-outer, what src/latte_physics.py:465-466 assumes):
#   tokens[   0: 256] = all 256 spatial patches at temporal slot 0  (gray)
#   tokens[1792:2048] = all 256 spatial patches at temporal slot 7  (white)
# Hypothesis B (spatial-outer, the transposed layout):
#   tokens[   0:   8] = 8 temporal slots at spatial patch 0
#   tokens[2040:2048] = 8 temporal slots at spatial patch 255

diff_temporal = (t[:256].mean(0) - t[1792:2048].mean(0)).abs().mean().item()
diff_spatial  = (t[:8].mean(0)   - t[2040:2048].mean(0)).abs().mean().item()
std_within_t_slot_hypA = t[:256].std(0).mean().item()   # low if hypA correct (all-gray patches similar)
std_within_t_slot_hypB = t[:8].std(0).mean().item()     # low if hypB correct (patch-0 across time close to zero if static)

print(f"|mean(first 256)  - mean(last 256)|  = {diff_temporal:.4f}   <- big => temporal-outer (hypA)")
print(f"|mean(first 8)    - mean(last 8)  |  = {diff_spatial:.4f}    <- big => spatial-outer  (hypB)")
print(f"std within first-256 block: {std_within_t_slot_hypA:.4f}")
print(f"std within first-8   block: {std_within_t_slot_hypB:.4f}")

if diff_temporal > 3 * diff_spatial:
    print("\nRESULT: temporal-outer layout.  reshape order in src/latte_physics.py:465-466 is CORRECT.")
elif diff_spatial > 3 * diff_temporal:
    print("\nRESULT: spatial-outer layout.  You MUST flip the reshape in src/latte_physics.py:465-466 before Phase 4/9.")
else:
    print("\nRESULT: inconclusive. Try longer/sharper synthetic contrast or print more indices.")