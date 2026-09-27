"""Nepal voter card number scanning via OCR.

Scope and honesty note: this is a RESEARCH SIMULATION. There is no live
government national-ID database to check the extracted number against, so
this module cannot and does not "verify" an ID card is genuine -- it only
extracts a printed ID number from a captured photo of a card (real or a
mock card printed for the study, per the consent form) well enough to use
as a second uniqueness key alongside face matching. Say this plainly in the
paper: OCR extraction accuracy is being evaluated, not government identity
verification.
"""

import os
import re
import shutil

import cv2
import numpy as np
import pytesseract

_DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
_COMMON_TESSERACT_PATHS = (
    "/opt/homebrew/bin/tesseract",
    "/usr/local/bin/tesseract",
    "/usr/bin/tesseract",
)


class OcrError(RuntimeError):
    pass


class TesseractUnavailable(OcrError):
    pass


class NoIdNumberFound(OcrError):
    pass


def _find_tesseract():
    configured_path = os.environ.get("TESSERACT_CMD")
    if configured_path:
        if os.path.isfile(configured_path) and os.access(configured_path, os.X_OK):
            return configured_path
        raise TesseractUnavailable(
            f"TESSERACT_CMD points to an unavailable executable: {configured_path}"
        )

    path_candidate = shutil.which("tesseract")
    if path_candidate:
        return path_candidate

    for candidate in _COMMON_TESSERACT_PATHS:
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


def _configure_tesseract():
    executable = _find_tesseract()
    if not executable:
        raise TesseractUnavailable(
            "Tesseract OCR is not installed or is not in PATH. "
            "Install it with `brew install tesseract`, then restart the application."
        )
    pytesseract.pytesseract.tesseract_cmd = executable
    return executable


def _preprocess(pil_image):
    gray = cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


def extract_id_number(pil_image):
    """Returns the voter number as ASCII digits, including Devanagari input,
    or raises NoIdNumberFound. Also returns the full raw OCR text in case the
    caller wants to display or log it."""
    _configure_tesseract()
    processed = _preprocess(pil_image)
    try:
        languages = set(pytesseract.get_languages(config=""))
        language = "+".join(name for name in ("eng", "nep", "hin") if name in languages)
        raw_text = pytesseract.image_to_string(processed, lang=language or "eng", config="--psm 6")
    except pytesseract.TesseractNotFoundError as exc:
        raise TesseractUnavailable(
            "Tesseract OCR could not be started. Check that `tesseract` is installed "
            "and available in PATH, then restart the application."
        ) from exc

    # Convert Nepali numerals and select the voter number from its labeled line.
    normalized_text = raw_text.translate(_DEVANAGARI_DIGITS)
    lines = normalized_text.splitlines()
    voter_lines = [line for line in lines if "मतदाता" in line]
    search_text = "\n".join(voter_lines) if voter_lines else normalized_text
    matches = re.findall(r"(?<!\d)\d(?:[\d\s./-]{4,24})\d(?!\d)", search_text)
    candidates = [re.sub(r"\D", "", value) for value in matches]
    candidates = [value for value in candidates if 6 <= len(value) <= 12]
    if not candidates:
        raise NoIdNumberFound("no ID-number-shaped text was found on the card")
    return candidates[0], raw_text.strip()


if __name__ == "__main__":
    try:
        executable = _configure_tesseract()
    except TesseractUnavailable as exc:
        print(f"ocr module unavailable: {exc}")
    else:
        print(f"ocr module ready: tesseract found at {executable}")
