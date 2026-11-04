"""
Utility / Geometry helpers used across multiple services.
Ported from telecom_utils.py — pure functions, zero dependencies on other app modules.
"""
from __future__ import annotations

import re
import math
from typing import Optional

import numpy as np


# ─────────────────────────────────────────────
#  Geometry
# ─────────────────────────────────────────────

def get_center(bbox: list[int]) -> tuple[int, int]:
    return ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)


def calculate_iou(box1: list[int], box2: list[int]) -> float:
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    xi_min = max(x1_min, x2_min)
    yi_min = max(y1_min, y2_min)
    xi_max = min(x1_max, x2_max)
    yi_max = min(y1_max, y2_max)

    if xi_max < xi_min or yi_max < yi_min:
        return 0.0

    intersection = (xi_max - xi_min) * (yi_max - yi_min)
    union = (
        (x1_max - x1_min) * (y1_max - y1_min)
        + (x2_max - x2_min) * (y2_max - y2_min)
        - intersection
    )
    return 0.0 if union == 0 else intersection / union


def calculate_distance(box1: list[int], box2: list[int]) -> float:
    c1 = get_center(box1)
    c2 = get_center(box2)
    return math.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)


def is_inside(obj_inner: dict, obj_outer: dict) -> bool:
    """True if obj_inner's bbox is fully contained within obj_outer's bbox."""
    b1, b2 = obj_inner["bbox"], obj_outer["bbox"]
    return b1[0] >= b2[0] and b1[1] >= b2[1] and b1[2] <= b2[2] and b1[3] <= b2[3]


def is_center_inside(obj_inner: dict, obj_outer: dict) -> bool:
    """True if obj_inner's center point falls within obj_outer's bbox."""
    cx, cy = get_center(obj_inner["bbox"])
    b2 = obj_outer["bbox"]
    return b2[0] <= cx <= b2[2] and b2[1] <= cy <= b2[3]


# ─────────────────────────────────────────────
#  Text Parsing
# ─────────────────────────────────────────────

def parse_power_data(text: str) -> tuple[Optional[int], Optional[float]]:
    """Extracts Voltage and Amperage from OCR text of a power supply."""
    if not text:
        return None, None
    volts_m = re.search(r"(\d+)\s*[vV]", text)
    amps_m = re.search(r"(\d+\.?\d*)\s*[aA]", text)
    volts = int(volts_m.group(1)) if volts_m else None
    amps = float(amps_m.group(1)) if amps_m else None
    return volts, amps


def parse_tap_value(text: str) -> str:
    if not text:
        return ""
    if "EQZ" in text.upper() or "CE" in text.upper():
        digits = re.findall(r"\d+", text)
        return f"EQZ-{digits[-1]}" if digits else "EQZ"
    return text


def clean_ocr_text(text: str, cls_name: str) -> str:
    """
    Class-aware OCR text cleaning — normalises common misreads,
    extracts relevant numeric/alphanumeric tokens per class.
    """
    if not text:
        return ""
    cls_lower = cls_name.lower()

    if "tap" in cls_lower:
        upper = text.upper()
        is_eqz = "EQZ" in upper or "CE" in upper
        text = text.translate(str.maketrans("oOlIzZsSgGqQbB", "00112255999966"))
        if is_eqz:
            digits = re.findall(r"\d+", text)
            prefix = "EQZ" if "EQZ" in upper else "CE"
            return f"{prefix}{digits[-1]}" if digits else prefix
        numbers = re.findall(r"\d+", text)
        valid = [n for n in numbers if len(n) <= 2]
        if valid:
            res = valid[-1]
            if len(res) == 2 and res[0] == res[1] and res[0] in "253689":
                res = res[0]
            return "0" if res == "00" else res
        return ""

    if "splitter" in cls_lower:
        text = text.translate(str.maketrans("oOlIzZsSgGbB", "001122559966"))
        numbers = re.findall(r"\d*\.?\d+", text)
        valid = [n for n in numbers if n != "." and float(n) != 0]
        if valid:
            res = valid[-1]
            return ("0" + res) if res.startswith(".") else res
        return ""

    if "power_supply" in cls_lower:
        m = re.search(r"(\d+)\s*[vV]", text)
        if m:
            return m.group(1)
        return ""  # NO FALLBACK: Strictly require 'V' (e.g., 60V, 90 V)

    if "tag_id" in cls_lower:
        text = text.translate(str.maketrans("oOlIzZsSgGbB", "001122559966"))
        tokens = [re.sub(r"[^a-zA-Z0-9-]", "", t) for t in text.split()]
        valid = [t for t in tokens if len(t) >= 1]
        return max(valid, key=len) if valid else ""

    # General fallback
    return " ".join(re.sub(r"[^a-zA-Z0-9\s.-]", "", text).split())


def deduplicate_global_callouts(callouts: list[dict], dist_thresh_px: float = 120.0) -> list[dict]:
    """
    Deduplicates callouts that were detected across adjacent overlapping tiles.
    1. Suppresses 'A' if 'B' (LE->Amp) or 'UPGRADE NODE' is within dist_thresh_px.
    2. Merges duplicate identical callouts within dist_thresh_px (keeps first/highest priority occurrence).
    """
    priority_order = {"UPGRADE NODE": 0, "B": 1, "G": 2, "J": 3, "E, ADD TERM": 4, "A": 10}
    sorted_callouts = sorted(callouts, key=lambda c: priority_order.get(c.get("text", "").strip(), 5))

    kept: list[dict] = []
    for c in sorted_callouts:
        c_text = c.get("text", "").strip()
        c_gx, c_gy = float(c.get("gx", 0)), float(c.get("gy", 0))

        is_dup = False
        for k in kept:
            k_text = k.get("text", "").strip()
            k_gx, k_gy = float(k.get("gx", 0)), float(k.get("gy", 0))
            dist = math.hypot(c_gx - k_gx, c_gy - k_gy)

            # Rule 1: Identical callouts within distance threshold (e.g. A vs A from overlapping tiles)
            if c_text == k_text and dist < dist_thresh_px:
                is_dup = True
                break

            # Rule 2: Suppress general 'A' if specific 'B' (LE->Amp) or 'UPGRADE NODE' is at the same symbol
            if c_text == "A" and (k_text == "B" or "NODE" in k_text) and dist < dist_thresh_px:
                is_dup = True
                break

        if not is_dup:
            kept.append(c)

    return kept
