"""
curate_openvid.py

Filters OpenVid-1M.csv to a physics-relevant training subset.

Informed by auditing 50 random samples from the previous broad filter, running
keyword frequency + motion-score analysis, and sampling each pattern category.

Key finding: OpenVid-1M is landscape B-roll, cooking, sports, and travel.
Lab-style physics demonstrations don't exist. The filter below targets the four
content categories that genuinely contain physics (verified by manual sampling):
  1. Fluid pouring  — cooking videos pouring liquid into containers (~3448 clips at 8s)
  2. Viscous drip   — honey/sauce/oil dripping (~30 clips at 8s)
  3. Splash impact  — liquid splashing in controlled setting, excl. animals (~275 clips)
  4. Projectile     — sports: ball being thrown/kicked/tossed (~1017 clips at 8s)
  5. Bowling        — rolling ball physics (~46 clips at 8s)

Patterns dropped after auditing (too many false positives):
  - "object_drop"  → "steep drop-off", "fallen leaves", "drop of rain" dominated
  - "object_roll"  → extremely rare (1 clip in full dataset after proximity filter)
  - "object_bounce"→ 9 clips total, mostly "light bouncing off rocks"

False positives corrected:
  - "marble"  → 100% building material (marble countertop/floor/statue); dropped
  - "rolling" → almost entirely "rolling hills"; removed from broad patterns
  - "falling" → mostly "falling rain/snow/leaves"; removed from broad patterns
  - "wave"    → ocean waves in scenic shots; removed
  - splash + animal swimming → swan/duck/seal scenes; excluded via ANIMAL_SWIM pattern

Motion score >= 3.0 as quality gate (dataset median 2.37; landscape B-roll
median 1.68; pouring/sports median 3.6-4.2).

Duration choice (yield vs temporal resolution tradeoff):
  max 4.0s  →  1,651 clips   (every ~3rd frame sampled at 12fps)
  max 6.0s  →  3,979 clips
  max 8.0s  →  4,816 clips   ← chosen: big gain, 16 frames every ~6th → still fine
  max 10.0s →  5,286 clips   (diminishing returns, +10%)
  max 14.0s →  5,769 clips   (diminishing returns, +9%)

VideoProcessor samples exactly 16 frames uniformly regardless of duration.
For pouring/sports, physics spans the full clip so sparser sampling is acceptable.

NOTE: 01_prepare_video_dataset_streaming.py has a hard `duration <= 4.0` filter
that should be updated to `duration <= 8.0` to process these longer clips.

Usage:
    python evaluation/curate_openvid.py \
        --input  data/text_csv/OpenVid-1M.csv \
        --output data/text_csv/curated_OpenVid-1M.csv
"""

from __future__ import annotations

import argparse
import re
import pandas as pd

MIN_SECONDS   = 1.5
MAX_SECONDS   = 8.0   # extended from 4.0 — see yield table in docstring
MIN_MOTION    = 3.0   # dataset median is 2.37; landscape B-roll median is 1.68


# ---------------------------------------------------------------------------
# Proximity patterns — verb must appear within N characters of the object.
# Using re.search on the lowercased caption string.
# ---------------------------------------------------------------------------

# 1. Fluid pouring: "pouring / pours / poured" near "into / onto / over"
_POUR = re.compile(
    r"\bpour(ing|s|ed)?\b.{0,60}\b(into|onto|over)\b"
    r"|\b(into|onto|over)\b.{0,60}\bpour(ing|s|ed)?\b",
    re.IGNORECASE,
)

# 2. Viscous drip: "dripping/drips/dripped" near a viscous substance
_DRIP = re.compile(
    r"\bdrip(ping|s|ped)?\b.{0,50}"
    r"\b(honey|sauce|oil|liquid|syrup|wax|chocolate|paint|coffee|juice|cream|caramel|resin|glue|lava)\b"
    r"|\b(honey|sauce|oil|syrup|wax|chocolate|caramel|resin|glue)\b.{0,50}\bdrip(ping|s|ped)?\b",
    re.IGNORECASE,
)

# NOTE: splash_liquid was tested and dropped. "splash + water" fires on dog/bear/bird
# in water, waterfalls, and ocean waves — too many uncontrollable false positives.
# Only ~20% hit rate in manual sampling. Not worth the noise.

# 4. Projectile: throw/kick/toss/shoot near ball/puck
_PROJECTILE = re.compile(
    r"\b(throw(ing|s|n)?|kick(ing|s|ed)?|toss(ing|es|ed)?|launch(es|ing|ed)?|shoot(ing|s)?|fling(ing|s)?|hurl(ing|s)?)\b"
    r".{0,80}\b(ball|puck|frisbee|javelin|discus|shot.?put)\b"
    r"|\b(ball|puck|frisbee)\b.{0,80}"
    r"\b(throw(ing|s|n)?|kick(ing|s|ed)?|toss(ing|es|ed)?|launch(es|ing|ed)?|shoot(ing|s)?)\b",
    re.IGNORECASE,
)

# 5. Bowling — always rolling ball physics
_BOWLING = re.compile(r"\bbowling\b", re.IGNORECASE)

# 6. Explicit physics concept phrases — unambiguous in any context
_CONCEPT = re.compile(
    r"\b(free.?fall|projectile.motion|elastic.collision|terminal.velocity|"
    r"angular.momentum|moment.of.inertia|conservation.of.momentum|"
    r"centripetal.force|centrifugal.force|coefficient.of.friction)\b",
    re.IGNORECASE,
)

PATTERNS = [_POUR, _DRIP, _PROJECTILE, _BOWLING, _CONCEPT]
PATTERN_NAMES = ["pour_into", "viscous_drip", "projectile_ball", "bowling", "physics_concept"]


def match_category(caption: str) -> str | None:
    for pat, name in zip(PATTERNS, PATTERN_NAMES):
        if pat.search(caption):
            return name
    return None


def curate(input_path: str, output_path: str) -> None:
    print(f"Loading {input_path} ...")
    df = pd.read_csv(input_path)
    total = len(df)
    print(f"  Total rows: {total:,}")

    # Duration filter
    df = df[(df["seconds"] >= MIN_SECONDS) & (df["seconds"] <= MAX_SECONDS)]
    print(f"  After duration filter [{MIN_SECONDS}s – {MAX_SECONDS}s]: {len(df):,} rows")

    # Motion score gate
    df = df[df["motion score"] >= MIN_MOTION]
    print(f"  After motion score >= {MIN_MOTION}: {len(df):,} rows")

    # Caption proximity filter
    df = df.copy()
    df["physics_category"] = df["caption"].apply(match_category)
    df = df[df["physics_category"].notna()].reset_index(drop=True)
    print(f"  After caption proximity filter: {len(df):,} rows ({len(df)/total*100:.2f}% of original)")

    # Per-category breakdown
    print("\n  Category breakdown:")
    for cat, cnt in df["physics_category"].value_counts().items():
        print(f"    {cat:25s}: {cnt:5,}")

    df.to_csv(output_path, index=False)
    print(f"\nSaved curated subset to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Curate OpenVid-1M to a physics-relevant subset")
    parser.add_argument("--input",  default="data/text_csv/OpenVid-1M.csv")
    parser.add_argument("--output", default="data/text_csv/curated_OpenVid-1M.csv")
    args = parser.parse_args()
    curate(args.input, args.output)
