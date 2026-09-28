"""OCR accuracy evaluation for the voter-card number step.

Put card photos in a folder and a ground-truth CSV next to them:

    cards/
      labels.csv      # columns: file,number   (number as ASCII digits)
      card_001.jpg ...

Usage:
    python evaluate_ocr.py cards/ --out ocr_results/

Reports exact-match accuracy (the number that matters: the whole ID must be
right), wrong-number rate (dangerous: silently stores another number),
no-result rate, digit-level accuracy, and timing.
"""

import argparse
import csv
import os
import time

from PIL import Image, ImageOps

from ocr import NoIdNumberFound, OcrError, extract_id_number_detailed, language_status


def levenshtein(a, b):
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder")
    parser.add_argument("--labels", default="labels.csv")
    parser.add_argument("--out", default="ocr_results")
    args = parser.parse_args()

    ok, message = language_status()
    print(("OCR: " if ok else "WARNING: ") + message)
    os.makedirs(args.out, exist_ok=True)

    with open(os.path.join(args.folder, args.labels), newline="", encoding="utf-8") as handle:
        labels = list(csv.DictReader(handle))

    rows, exact, wrong, none, times, digit_errors, digit_total = [], 0, 0, 0, [], 0, 0
    for item in labels:
        truth = "".join(ch for ch in item["number"] if ch.isdigit())
        with Image.open(os.path.join(args.folder, item["file"])) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
        started = time.perf_counter()
        try:
            result = extract_id_number_detailed(image)
            predicted = result["number"]
        except (NoIdNumberFound, OcrError):
            predicted = ""
        times.append(time.perf_counter() - started)
        if predicted == truth:
            outcome = "exact"; exact += 1
        elif predicted:
            outcome = "wrong"; wrong += 1
        else:
            outcome = "no_result"; none += 1
        digit_errors += levenshtein(predicted, truth)
        digit_total += len(truth)
        rows.append((item["file"], truth, predicted, outcome, f"{times[-1]:.2f}"))

    n = len(labels)
    summary = "\n".join([
        f"cards: {n}",
        f"exact match: {exact} ({exact / n * 100:.1f}%)",
        f"wrong number returned: {wrong} ({wrong / n * 100:.1f}%)",
        f"no number found: {none} ({none / n * 100:.1f}%)",
        f"digit-level accuracy (1 - CER): {(1 - digit_errors / max(digit_total, 1)) * 100:.2f}%",
        f"time per card: mean {sum(times) / n:.2f} s, max {max(times):.2f} s",
    ])
    print(summary)
    with open(os.path.join(args.out, "ocr_results.csv"), "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["file", "truth", "predicted", "outcome", "seconds"])
        writer.writerows(rows)
    with open(os.path.join(args.out, "summary.txt"), "w") as handle:
        handle.write(summary + "\n")


if __name__ == "__main__":
    main()
