"""Face-verification accuracy evaluation (FAR / FRR / EER / ROC) for the paper.

Dataset layout (photos of consenting volunteers, taken with the SAME kind of
camera and lighting used in the study):

    dataset/
      person01/  enroll_1.jpg enroll_2.jpg enroll_3.jpg  probe_1.jpg probe_2.jpg ...
      person02/  ...

Protocol (mirrors the app exactly):
  * Enrollment template = mean of the first --enroll images of each person
    (sorted by file name), exactly like the 3-photo registration.
  * Every remaining image of the person is a GENUINE probe against their own
    template; every image of every OTHER person is an IMPOSTOR probe.
  * Failure-to-enroll (FTE) and failure-to-acquire (FTA: no/multiple/low-
    quality face on a probe) are reported separately, as ISO/IEC 19795-1
    recommends. They are NOT silently dropped.

Usage:
    python evaluate_accuracy.py dataset/ --enroll 3 --out results/
Outputs: results/scores.csv, results/summary.txt, results/roc.png (if matplotlib)
"""

import argparse
import csv
import os
import sys
import time

import numpy as np
from PIL import Image, ImageOps

from face_engine import SFACE_MATCH_THRESHOLD, FaceError, build_registration_template, ensure_models, get_engine

IMAGE_EXT = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def load(path):
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def rates_at(genuine, impostor, threshold):
    frr = float(np.mean(genuine < threshold)) if len(genuine) else float("nan")
    far = float(np.mean(impostor >= threshold)) if len(impostor) else float("nan")
    return far, frr


def eer(genuine, impostor):
    thresholds = np.unique(np.concatenate([genuine, impostor]))
    best = (1.0, None, None, None)
    for t in thresholds:
        far, frr = rates_at(genuine, impostor, t)
        gap = abs(far - frr)
        if gap < best[0]:
            best = (gap, t, far, frr)
    _, t, far, frr = best
    return (far + frr) / 2.0, t


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset")
    parser.add_argument("--enroll", type=int, default=3, help="images per person used for enrollment")
    parser.add_argument("--out", default="evaluation_results")
    args = parser.parse_args()

    ensure_models(download=True)
    engine = get_engine()
    os.makedirs(args.out, exist_ok=True)

    people = sorted(d for d in os.listdir(args.dataset) if os.path.isdir(os.path.join(args.dataset, d)))
    templates, probes = {}, []
    fte, fta, fta_reasons, times = [], 0, {}, []
    for person in people:
        files = sorted(
            os.path.join(args.dataset, person, f)
            for f in os.listdir(os.path.join(args.dataset, person))
            if f.lower().endswith(IMAGE_EXT)
        )
        if len(files) <= args.enroll:
            print(f"skip {person}: needs more than {args.enroll} images", file=sys.stderr)
            continue
        try:
            samples = [engine.extract(load(f)) for f in files[: args.enroll]]
            template, _, consistency = build_registration_template(samples)
            templates[person] = np.asarray(template)
        except FaceError as exc:
            fte.append((person, str(exc)))
            continue
        for f in files[args.enroll:]:
            started = time.perf_counter()
            try:
                sample = engine.extract(load(f))
                times.append((time.perf_counter() - started) * 1000.0)
                probes.append((person, f, sample.embedding))
            except FaceError as exc:
                fta += 1
                fta_reasons[type(exc).__name__] = fta_reasons.get(type(exc).__name__, 0) + 1

    genuine, impostor, rows = [], [], []
    for person, path, embedding in probes:
        for claimed, template in templates.items():
            score = float(np.dot(embedding, template))
            kind = "genuine" if claimed == person else "impostor"
            (genuine if kind == "genuine" else impostor).append(score)
            rows.append((path, person, claimed, kind, f"{score:.6f}"))
    genuine, impostor = np.array(genuine), np.array(impostor)

    with open(os.path.join(args.out, "scores.csv"), "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["probe", "true_identity", "claimed_identity", "type", "cosine_score"])
        writer.writerows(rows)

    lines = [
        f"people enrolled: {len(templates)}  failure-to-enroll: {len(fte)}",
        f"probe images: {len(probes) + fta}  failure-to-acquire: {fta} {fta_reasons}",
        f"genuine comparisons: {len(genuine)}  impostor comparisons: {len(impostor)}",
    ]
    if len(genuine) and len(impostor):
        e, t_eer = eer(genuine, impostor)
        lines.append(f"genuine score mean {genuine.mean():.4f} (min {genuine.min():.4f}); "
                     f"impostor mean {impostor.mean():.4f} (max {impostor.max():.4f})")
        lines.append(f"EER = {e * 100:.2f}% at threshold {t_eer:.4f}")
        lines.append("threshold   FAR       FRR")
        for t in sorted({0.25, 0.30, SFACE_MATCH_THRESHOLD, 0.40, 0.45, 0.50, round(float(t_eer), 4)}):
            far, frr = rates_at(genuine, impostor, t)
            lines.append(f"{t:9.4f}  {far * 100:7.3f}%  {frr * 100:7.3f}%" + ("   <- app threshold" if abs(t - SFACE_MATCH_THRESHOLD) < 1e-9 else ""))
        for target in (0.01, 0.001):
            t = float(np.quantile(impostor, 1 - target)) if len(impostor) else float("nan")
            _, frr = rates_at(genuine, impostor, t)
            lines.append(f"TAR @ FAR={target * 100:g}%: {(1 - frr) * 100:.2f}% (threshold {t:.4f})")
        if len(impostor) < 1000:
            lines.append("NOTE: fewer than 1000 impostor comparisons -- FAR below ~0.1% cannot be measured; say so in the paper.")
    if times:
        lines.append(f"extract time per image: median {np.median(times):.1f} ms, p95 {np.percentile(times, 95):.1f} ms")
    if fte:
        lines.append("failure-to-enroll details: " + "; ".join(f"{p}: {r}" for p, r in fte))

    summary = "\n".join(lines)
    print(summary)
    with open(os.path.join(args.out, "summary.txt"), "w") as handle:
        handle.write(summary + "\n")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        thresholds = np.linspace(-0.2, 1.0, 500)
        fars = [rates_at(genuine, impostor, t)[0] for t in thresholds]
        frrs = [rates_at(genuine, impostor, t)[1] for t in thresholds]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].hist(impostor, bins=50, alpha=0.6, label="impostor", density=True)
        axes[0].hist(genuine, bins=50, alpha=0.6, label="genuine", density=True)
        axes[0].axvline(SFACE_MATCH_THRESHOLD, color="k", linestyle="--", label="threshold")
        axes[0].set_xlabel("cosine similarity"); axes[0].legend(); axes[0].set_title("Score distributions")
        axes[1].plot(fars, [1 - f for f in frrs])
        axes[1].set_xscale("log"); axes[1].set_xlabel("FAR"); axes[1].set_ylabel("TAR (1 - FRR)"); axes[1].set_title("ROC")
        fig.tight_layout(); fig.savefig(os.path.join(args.out, "roc.png"), dpi=200)
        print(f"plot written to {os.path.join(args.out, 'roc.png')}")
    except ImportError:
        print("(install matplotlib for roc.png)")


if __name__ == "__main__":
    main()
