# 🐛 QC Bug Report — Telecom Vision AI Tool
**Project:** Telecom Vision API (Docker-Tool)  
**Date:** 2026-07-30  
**Reported By:** QC / Testing Team (22 testers, 40+ PIDs tested)  
**Status:** Open — Awaiting Remediation  
**Data Source:** AI review.xlsx (COAX sheet + FIBER sheet + SUGGESTIONS)

---

## 📊 Real Accuracy Numbers from QC Testing (COAX Sheet)

> Format in Excel: `Detected / Actual` — e.g., `14/22` means model found 14 out of 22 real objects.

| Callout Type | Detected | Actual | **Accuracy** | Status |
|---|---|---|---|---|
| **A** (Amplifier) | 353 | 1,472 | **24.0%** | 🔴 Critical |
| **ADD CE–XX** (TAP/CEX) | 0 | 72 | **0.0%** | 🔴 Critical — completely broken |
| **REMOVE TERM** | 5 | 30 | **16.7%** | 🔴 Critical |
| **ADD TERM** | 41 | 127 | **32.3%** | 🔴 High |
| **E** (Tap value change) | 201 | 568 | **35.4%** | 🔴 High |
| **ADD SPLICE** | 21 | 53 | **39.6%** | 🟠 Medium |
| **J** (Equalizer removed) | 191 | 354 | **54.0%** | 🟠 Medium |
| **UPGRADE NODE** | 40 | 73 | **54.8%** | 🟠 Medium |
| **UPGRADE INT 2WAY** | 55 | 96 | **57.3%** | 🟠 Medium |
| **UPGRADE POWER SUPPLY** | 12 | 19 | **63.2%** | 🟡 Acceptable |
| **UPGRADE INT DC** | 59 | 97 | **60.8%** | 🟡 Acceptable |
| **H** (New LE/Booster) | 70 | 105 | **66.7%** | 🟡 Acceptable |
| **G** (Splitter change) | 33 | 41 | **80.5%** | 🟢 OK |

### Accuracy by DPI/Resolution
| DPI | Detected | Total | Accuracy |
|-----|----------|-------|----------|
| 300 | 173 | 550 | **31.5%** |
| 600 | 624 | 1,549 | **40.3%** ← Best |
| 800 | 353 | 1,008 | **35.0%** |

> **Key insight:** 600 DPI gives the best overall results. 300 DPI performs worst. 800 DPI adds noise.

---

## 📋 Fiber Sheet Issues (Qualitative — 57 test rows)

Issues reported across all fiber PIDs (DPI 50–90):

| Issue | Frequency |
|---|---|
| Callout overlapping | 🔴 **100% of cases** |
| Text box color changes to black when moved | ~70% of cases |
| Cannot move text boxes | ~60% of cases |
| Callout going outside print border | ~25% of cases |
| Callout arrow not pointing correctly | ~25% of cases |
| Splice #1 / #2 missing | ~15% of cases |
| Duplicate callouts | ~15% of cases |
| Hub callouts beyond border | ~10% of cases |
| Overview moved too far | Seen in PID 6612485 |

---

## 💬 Tester Suggestions (SUGGESTIONS Sheet — 22 Testers)

Consolidated from all tester feedback:

1. **"Callouts overlapping the equipment — not pointed correctly"** ← Most common complaint
2. **"Callouts for MUX LOCATION and SPLICE #2 are missing"** — Reported by multiple testers
3. **"Most AMP callouts are missing — many unnecessary callouts added"** (Abhijith C)
4. **"When moving callout, border turns black"** — Red border breaking on move
5. **"Prism screenshot is on top of equipment in the print"** — Survey image overlap
6. **"Images and callouts can't adjust/move"**
7. **"After fiber callout has wrong border colour"**
8. **"Missing option for 'upgrade to 6ct'"** (Nobil Babu) — Feature gap
9. **"Title block out of place"**
10. **"Textbox order is wrong"** (Aiswarya MG)
11. **"Total house count callout not added in 2×2"** (Abhijith C)
12. **"Need text to be automatically capitalised"** (Nobil Babu)

---

## 🔍 Issue Breakdown — Updated with Real Data

---

### Issue #1 — Callout A (Amplifier) — 24% Accuracy 🔴 CRITICAL

**Severity:** 🔴 Critical  
**Data:** 353 detected out of 1,472 actual → **76% of amplifiers are missed**

**Root Cause:**
- Model `best.pt` significantly under-detects amplifiers — likely due to class imbalance in training data
- Amplifiers visually resemble Line Extenders; the model confuses or skips them
- "Most AMP callouts are missing, many unnecessary callouts added" (tester feedback)

**Fix:**
1. Retrain `best.pt` with heavily augmented amplifier samples
2. Lower confidence threshold specifically for class `A`
3. Add hard-negative mining for amplifier lookalikes

**Effort:** 1 day (threshold) + 4–6 weeks (retrain)

---

### Issue #2 — ADD CE–XX (TAP/CEX) — 0% Accuracy 🔴 COMPLETELY BROKEN

**Severity:** 🔴 Critical  
**Data:** 0 detected out of 72 actual → **model has never successfully detected this class**

**Root Cause:**
- `best.pt` was either not trained on TAP/CEX classes, or they were mislabelled as JP
- This is a data gap, not a tuning problem — model has zero ability to detect these

**Fix:**
1. Collect and annotate TAP/CEX examples from real maps (minimum 200–500 samples)
2. Retrain `best.pt` with TAP and CEX as explicit distinct classes
3. Cannot be solved any other way — retraining is mandatory

**Effort:** 3–5 weeks (data collection + annotation + retrain)

---

### Issue #3 — REMOVE TERM — 16.7% Accuracy 🔴 CRITICAL

**Severity:** 🔴 High  
**Data:** 5 detected out of 30 actual → **83% of terminator removals are missed**

**Root Cause:**
- Terminator removal is handled by matching — if the Before-map terminator isn't matched, the removal is never detected
- Small symbol size makes terminators easy to miss in tiled inference

**Fix:**
1. Extend rules engine to handle unmatched Before-only detections for terminators
2. Improve terminator detection confidence in `best.pt`

**Effort:** 1–2 days (rules fix) + 2–3 weeks (model improvement)

---

### Issue #4 — Callout E (Tap Value Change) — 35.4% Accuracy 🔴 HIGH

**Severity:** 🔴 High  
**Data:** 201 detected out of 568 actual → **64.6% of tap changes are missed**

**Root Cause:**
- EasyOCR frequently fails to read the numeric value inside tap symbols
- When OCR fails, the rule silently skips the callout
- Low DPI or small tap size exacerbates this

**Fix:**
1. Add OCR fallback — if OCR fails but tap detected in After and not matched in Before, emit `E (value unreadable)`
2. Switch to PaddleOCR for better small-number accuracy
3. Retrain with more diverse tap samples

**Effort:** 2–3 days (fallback) + 2–4 weeks (retrain)

---

### Issue #5 — Callout Overlapping (100% of Fiber Cases) 🔴 CRITICAL

**Severity:** 🔴 Critical  
**Sheets Affected:** FIBER (100% of rows), COAX (majority of rows)

**Root Cause:**
- Collision avoidance in `reporting.py` fails on dense maps
- Callouts placed on top of equipment symbols
- Survey Info (Prism screenshot) overlaps map equipment

**Fix:**
1. Rewrite placement engine — use grid-based spatial indexing to avoid equipment bounding boxes, not just other callouts
2. Add equipment-exclusion zones to collision avoidance
3. Implement a "push-out" algorithm that moves callouts to the nearest empty region

**Effort:** 3–5 days

---

### Issue #6 — Text Box Color Changes to Black When Moved 🟠 MEDIUM

**Severity:** 🟠 Medium  
**Frequency:** ~70% of all fiber test cases  
**Testers:** Ann Pearl Shaju, Akhil NA, Nobil Babu, and others

**Root Cause:**
- When a FreeText annotation is moved in Adobe Acrobat/Foxit, the viewer applies its default style (black border) if the annotation's appearance stream is not embedded correctly
- PyMuPDF's `add_freetext_annot()` may not be embedding the `/AP` (appearance stream) entry, causing viewers to re-render with default styles on move

**Fix:**
1. After creating each annotation, call `annot.update()` with `opacity=1` to force appearance stream generation in PyMuPDF
2. Alternatively set the `/MK` annotation appearance entry manually
3. Test in Adobe Acrobat Reader (free) and Bluebeam

**Effort:** 1–2 days

---

### Issue #7 — Cannot Move Text Boxes 🟠 MEDIUM

**Severity:** 🟠 Medium  
**Frequency:** ~60% of fiber test cases

**Root Cause:**
- Annotations may have the `Lock` flag set, or PDF viewer being used (browser PDF viewer) doesn't support annotation movement
- `add_freetext_annot()` default flags may be locking the annotation

**Fix:**
1. Explicitly set annotation flags: `annot.set_flags(4)` (Print only, not locked) in PyMuPDF
2. Document to users: must use Adobe Acrobat or Foxit Reader, not browser PDF viewer

**Effort:** 0.5 day

---

### Issue #8 — Callout Arrow Points Incorrectly 🟠 MEDIUM

**Severity:** 🟠 Medium  
**Frequency:** ~25% of fiber test cases; also in COAX

**Root Cause:**
- Coordinate transformation from tile-pixel space to PDF page space has an offset bug
- PDF Y-axis is inverted vs. image Y-axis; tile offset may not be correctly added to the final PDF coordinate

**Fix:**
1. Audit coordinate pipeline in `reporting.py` — log first 5 callout positions and manually verify
2. Add a unit test: assert arrow tip falls within bounding box of detected object

**Effort:** 1–3 days

---

### Issue #9 — Callouts Going Outside Print Border 🟠 MEDIUM

**Severity:** 🟠 Medium  
**Frequency:** ~25% of test cases

**Root Cause:**
- Callout placement engine places callouts without checking the page boundary
- On edge-of-map objects, the callout text box extends beyond the printable area

**Fix:**
1. Add boundary clipping in `reporting.py` — clamp callout rectangle to `page.rect` minus a margin
2. If no in-bounds space found, prioritize inward placement direction

**Effort:** 1 day

---

### Issue #10 — Duplicate Callouts 🟡 LOW–MEDIUM

**Severity:** 🟡 Low–Medium  
**Frequency:** ~15% of test cases

**Root Cause:**
- Deduplication radius of 50 PDF points is too small
- Two nearly-identical detections (same object detected in two overlapping tiles) produce two callouts

**Fix:**
1. Increase deduplication radius to 80–100 PDF points
2. Also deduplicate by class prefix (not just exact text match)

**Effort:** 0.5–1 day

---

### Issue #11 — Splice #1 / #2 Missing 🔴 HIGH

**Severity:** 🔴 High  
**Frequency:** ~15% of fiber test cases; some cases show Extra Splice #1

**Root Cause (Missing):**
- Splice callout generation in fiber pipeline doesn't handle all splice configurations
- Splice #2 appears to be completely absent from some PIDs

**Root Cause (Extra):**
- Fiber pipeline may be misidentifying another object as Splice #1

**Fix:**
1. Audit fiber pipeline splice detection rules — confirm both `SPLICE #1` and `SPLICE #2` logic paths exist and are triggered correctly
2. Add validation: if a splice is present in Before but not in After, flag it

**Effort:** 1–2 days

---

### Issue #12 — Hub / Port Callouts Beyond Print Border 🟠 MEDIUM

**Severity:** 🟠 Medium  
**PIDs Affected:** 4419742, 4421238

**Root Cause:**
- Same as Issue #9 — no boundary checking in fiber callout placement

**Fix:** Same as Issue #9 fix (boundary clamping)

**Effort:** Covered by Issue #9 fix

---

### Issue #13 — Missing Feature: "Upgrade to 6CT" Option 🟡 FEATURE GAP

**Severity:** 🟡 Low (Feature Gap)  
**Reported by:** Nobil Babu

**Description:** No callout option exists for "upgrade to 6ct" fiber scenario.

**Fix:** Add new callout rule in rules engine + new label in frontend dropdown

**Effort:** 1 day

---

### Issue #14 — "Overview Moved Too Far" in Fiber 🟡 LOW

**Severity:** 🟡 Low  
**PID:** 6612485 (DPI 70 and 90)

**Root Cause:**
- The survey info overlay position calculation in `fiber_overview.py` computes an offset that is too large at higher DPIs

**Fix:**
- Normalize overlay position by DPI ratio in `fiber_overview.py`

**Effort:** 0.5 day

---

## 📊 Summary Table

| # | Issue | Accuracy / Freq | Severity | Fix Time |
|---|-------|-----------------|----------|----------|
| 1 | Amplifier (A) missed | **24%** | 🔴 Critical | 1 day + 4–6 wk retrain |
| 2 | TAP/CEX (ADD CE–XX) not detected | **0%** | 🔴 Critical | 3–5 wk retrain |
| 3 | REMOVE TERM missed | **16.7%** | 🔴 High | 1–2 days + retrain |
| 4 | Tap value E missed | **35.4%** | 🔴 High | 2–3 days + 2–4 wk retrain |
| 5 | Callout overlapping | **100% of fiber** | 🔴 Critical | 3–5 days |
| 6 | Border turns black on move | ~70% of fiber | 🟠 Medium | 1–2 days |
| 7 | Cannot move text boxes | ~60% of fiber | 🟠 Medium | 0.5 day |
| 8 | Arrow points wrong location | ~25% cases | 🟠 Medium | 1–3 days |
| 9 | Callouts outside print border | ~25% cases | 🟠 Medium | 1 day |
| 10 | Duplicate callouts | ~15% cases | 🟡 Low–Med | 0.5–1 day |
| 11 | Splice #1/#2 missing/extra | ~15% fiber cases | 🔴 High | 1–2 days |
| 12 | Hub callouts beyond border | PID-specific | 🟠 Medium | (covered by #9) |
| 13 | Missing "Upgrade to 6CT" | Feature gap | 🟡 Low | 1 day |
| 14 | Overview moved too far | DPI 70–90 | 🟡 Low | 0.5 day |

---

## 🕐 Recommended Fix Timeline

### Sprint 1 — Week 1–2 (Code Fixes, No Retraining)
*Fixes that don't need new model weights*

| Task | Fixes | Days |
|------|-------|------|
| Fix annotation flags (movability + color) | #6, #7 | 1 day |
| Fix callout boundary clamping | #9, #12 | 1 day |
| Fix arrow coordinate pipeline | #8 | 1–3 days |
| Increase deduplication radius | #10 | 0.5 day |
| Fix REMOVE TERM in rules engine | #3, #11 | 1–2 days |
| Fix overview placement at high DPI | #14 | 0.5 day |
| Add "Upgrade to 6CT" callout | #13 | 1 day |
| Rework callout placement / overlap | #5 | 3–5 days |

**Sprint 1 Total: ~1.5–2 weeks, 1–2 backend devs**

---

### Sprint 2 — Week 2–4 (Quick Model Tuning)
*Without full retraining*

| Task | Fixes | Days |
|------|-------|------|
| Lower confidence threshold for class A | #1 | 1 day |
| Add OCR fallback for E callout | #4 | 2–3 days |
| Add E2E regression tests | All | 3–5 days |
| Extend matching for unmatched removals | #3, #11 | 1–2 days |

**Sprint 2 Total: ~2 weeks, 1 backend + 1 ML engineer**

---

### Sprint 3 — Week 4–10 (Model Retraining)
*Mandatory for Issues #1, #2, #4*

| Task | Target |
|------|--------|
| Collect & annotate TAP/CEX examples (200–500 samples) | #2 |
| Collect & annotate Amplifier hard negatives | #1 |
| Collect tap symbols at all DPIs and styles | #4 |
| Retrain `best.pt` with balanced class data | #1, #2, #4 |
| Validate: target mAP ≥ 0.85 per class | All |
| A/B test new vs old weights on QC test PIDs | All |
| Deploy new model weights | |

**Sprint 3 Total: 4–6 weeks, 1–2 ML engineers + QC annotation team**

---

## 🤔 On Retraining — Is It Necessary?

| Issue | Needs Retraining? | Notes |
|-------|------------------|-------|
| A (24% accuracy) | ✅ Yes (long-term) | Threshold tuning helps short-term |
| ADD CE–XX (0% accuracy) | ✅ **Mandatory** | Cannot be fixed any other way |
| E (35% accuracy) | ✅ Yes (partial) | OCR fallback helps short-term |
| REMOVE TERM | ⚠️ Partial | Rules fix first, then retrain |
| Overlapping callouts | ❌ No | Pure code fix in `reporting.py` |
| Text box color / movement | ❌ No | PyMuPDF annotation flag fix |
| Arrow direction | ❌ No | Coordinate transform bug fix |
| Splice missing | ❌ No | Rules engine fix |
| Overview too far | ❌ No | DPI normalization fix |

**Bottom line: 10 of 14 issues can be fixed without retraining. Retraining is mandatory only for #1 (A), #2 (TAP/CEX), and #4 (E) — and should be done together in one training run.**

---

## 📌 Recommended Optimal DPI

Based on the data: **Use 600 DPI** (40.3% accuracy vs 31.5% at 300 DPI and 35% at 800 DPI).  
Consider locking DPI to 600 in the API until multi-DPI training is available.
