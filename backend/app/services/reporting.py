"""
Reporting Service — Restored to match Original Reference Design
----------------------------------------------------------------
- Callouts: 10pt Helvetica, yellow fill, red border, arrow pointing at object
- Title box: auto-sized, yellow fill, red border, insert_textbox with proper lines
- Keeps font_size parameter so Coax (24pt box), Fiber Overview (14pt box),
  Fiber After (34pt box) all remain in title box only; callout text is always 10pt.
"""
from __future__ import annotations
import logging, math, fitz, numpy as np, re, cv2
from pathlib import Path
from app.services.alignment import pdf_to_image

def _patch_annot_color(doc: fitz.Document, annot: fitz.Annot, font_size: int = 10, is_callout: bool = True) -> None:
    """Patches annotation dictionary keys directly via xref so MuPDF does not mark annotations as dirty."""
    try:
        border_w = "1" if is_callout else "2.5"
        doc.xref_set_key(annot.xref, "F", "4") # fitz.PDF_ANNOT_IS_PRINT
        doc.xref_set_key(annot.xref, "BS", f"<< /W {border_w} /S /S >>") # 1pt for callouts, 2.5pt for title box
        doc.xref_set_key(annot.xref, "C", "[1 0 0]") # Red stroke
        doc.xref_set_key(annot.xref, "IC", "[1 1 0]") # Yellow fill
        doc.xref_set_key(annot.xref, "DA", f"(0 0 0 rg /Helv {font_size} Tf)")
        doc.xref_set_key(
            annot.xref, "DS",
            f"(font: Helv {font_size}pt; color: #000000; background-color: #FFFF00; border: {border_w}pt solid #FF0000;)"
        )
        
        if is_callout:
            doc.xref_set_key(annot.xref, "IT", "/FreeTextCallout")
            doc.xref_set_key(annot.xref, "LE", "[/OpenArrow /None]")
        else:
            doc.xref_set_key(annot.xref, "IT", "/FreeText")
    except Exception as e:
        logger.warning(f"Error patching annotation {annot.xref}: {e}")


def _stamp_movable_image(page: fitz.Page, rect: fitz.Rect, img_path: Path | str) -> bool:
    """
    Stamps an image as an interactive, movable and selectable PDF Stamp Annotation (/Stamp).
    This allows users in Adobe Acrobat or Bluebeam to click, move, resize, or delete the screenshot.
    """
    try:
        doc = page.parent
        with open(str(img_path), "rb") as f:
            img_bytes = f.read()

        annot = page.add_stamp_annot(rect)
        annot.set_rect(rect)
        annot.set_flags(4)  # fitz.PDF_ANNOT_IS_PRINT
        annot.update()

        # Insert image into document off-screen to generate an image XObject xref
        img_xref = page.insert_image(fitz.Rect(-5000, -5000, -4900, -4850), stream=img_bytes)

        w = rect.width
        h = rect.height
        form_stream = f"q {w:.2f} 0 0 {h:.2f} 0 0 cm /Im0 Do Q\n".encode("ascii")

        form_xref = doc.get_new_xref()
        form_dict = f"""<<
  /Type /XObject
  /Subtype /Form
  /BBox [ 0 0 {w:.2f} {h:.2f} ]
  /Matrix [ 1 0 0 1 0 0 ]
  /Resources <<
    /XObject <<
      /Im0 {img_xref} 0 R
    >>
  >>
>>"""
        doc.update_object(form_xref, form_dict)
        doc.update_stream(form_xref, form_stream)
        doc.xref_set_key(annot.xref, "AP", f"<< /N {form_xref} 0 R >>")
        return True
    except Exception as e:
        logger.warning(f"Could not stamp movable image: {e}, falling back to direct page insert")
        try:
            page.insert_image(rect, filename=str(img_path))
            return True
        except Exception as e2:
            logger.error(f"Failed to insert image: {e2}")
            return False

logger = logging.getLogger(__name__)

LEGEND_DATA: dict[str, str] = {
    "A": "SWAP ACTIVE",
    "B": "SWAP LE WITH AMP",
    "C": "MOVE ACTIVE",
    "D": "TAP FPC",
    "E": "TAP NEW",
    "F": "SPLITTER FPC",
    "G": "SPLITTER NEW",
    "H": "ADD ACTIVE",
    "I": "EQ NEW",
    "J": "EQ REMOVE",
    "K": "EQ FPC",
}

# ─────────────────────────────────────────────────────────────────────────────
# Public: Main vector report generator (Coax + Fiber After)
# ─────────────────────────────────────────────────────────────────────────────

def generate_vector_report(
    after_pdf_path: str | Path,
    callout_records: list[dict],
    tile_offsets: dict,          # kept for API compat; not used (gx/gy preferred)
    W_inv: np.ndarray,           # kept for API compat
    output_path: str | Path,
    dpi: int = 300,
    survey_image_path: str | Path | None = None,
    title_box_data: dict | None = None,
    title_font_size: int = 24,   # 24 Coax | 14 Fiber Overview | 34 Fiber After
    include_legend: bool = True,
) -> Path:
    after_pdf_path = Path(after_pdf_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Override font size based on map type ──────────────────────────────────
    if title_box_data:
        m_type = title_box_data.get("map_type", "").upper()
        if "OVERVIEW" in m_type:
            title_font_size = 14
        elif "FIBER" in m_type or "SCHEMATIC" in m_type:
            title_font_size = 34
        elif "COAX" in m_type:
            title_font_size = 24

    logger.info(f"Loading after PDF image at {dpi} DPI for empty-space detection...")
    img_after = pdf_to_image(after_pdf_path, dpi=dpi)
    img_gray = cv2.cvtColor(img_after, cv2.COLOR_BGR2GRAY)
    h_img, w_img = img_gray.shape

    doc = fitz.open(str(after_pdf_path))
    page = doc[0]
    placed_rects: list[fitz.Rect] = []
    hard_rects: list[fitz.Rect] = []

    # ── Reserve title-box area so callouts avoid it ───────────────────────────
    page_rect = page.rect
    title_reserve = fitz.Rect(page_rect.width - 720, 0, page_rect.width, 350)
    placed_rects.append(title_reserve)
    hard_rects.append(title_reserve)

    # ── Render callout annotations ────────────────────────────────────────────
    if callout_records:
        CALLOUT_FONT_SIZE = 10          # 10pt Helvetica as per specification
        DEDUP_RADIUS_PTS  = 80.0        # Merge identical labels within 80pt
        
        # Arrow offset: coax (high DPI >= 300) works best with 30, fiber (low DPI <= 90) needs 65+
        if dpi < 100:
            ARROW_OFFSET_PTS = 65.0
        elif dpi < 200:
            ARROW_OFFSET_PTS = 45.0
        else:
            ARROW_OFFSET_PTS = 30.0

        SEARCH_RADIUS_MAX = 600         # Max spiral search for free space

        unique: list[dict] = []
        for rec in callout_records:
            # Prefer global coords (gx/gy) — fallback to lx/ly (tile-relative)
            if "gx" in rec and "gy" in rec:
                gx, gy = float(rec["gx"]), float(rec["gy"])
                pt = W_inv @ np.array([gx, gy, 1.0])
                img_x, img_y = pt[0] / pt[2], pt[1] / pt[2]
            else:
                # Legacy tile-relative → global via W_inv transform
                tile_idx = rec.get("tile_idx", 0)
                ox, oy = tile_offsets.get(tile_idx, (0, 0))
                gx, gy = ox + float(rec.get("lx", 0)), oy + float(rec.get("ly", 0))
                pt = W_inv @ np.array([gx, gy, 1.0])
                img_x, img_y = pt[0] / pt[2], pt[1] / pt[2]

            pdf_x = img_x * (72.0 / dpi)
            pdf_y = img_y * (72.0 / dpi)
            text  = rec.get("text", "")

            # Dedup & conflict suppression
            has_b_or_node = any(
                (c["text"] == "B" or "NODE" in c["text"]) and
                math.hypot(pdf_x - c["pdf_x"], pdf_y - c["pdf_y"]) < DEDUP_RADIUS_PTS
                for c in unique
            )
            if text == "A" and has_b_or_node:
                continue

            if not any(
                c["text"] == text and
                math.hypot(pdf_x - c["pdf_x"], pdf_y - c["pdf_y"]) < DEDUP_RADIUS_PTS
                for c in unique
            ):
                unique.append({
                    "pdf_x": pdf_x, 
                    "pdf_y": pdf_y, 
                    "text": text, 
                    "is_design_note": rec.get("is_design_note", False)
                })

        for c in unique:
            cx_pdf, cy_pdf = c["pdf_x"], c["pdf_y"]
            text = c["text"]
            
            # The reference logic explicitly required img_x and img_y
            cx_img, cy_img = cx_pdf * (dpi / 72.0), cy_pdf * (dpi / 72.0)

            # ── Exclusion zone: reserve 35pt radius around the symbol ─────────
            SYMBOL_GUARD_PTS = 35.0
            symbol_guard = fitz.Rect(
                cx_pdf - SYMBOL_GUARD_PTS, cy_pdf - SYMBOL_GUARD_PTS,
                cx_pdf + SYMBOL_GUARD_PTS, cy_pdf + SYMBOL_GUARD_PTS,
            )
            placed_rects.append(symbol_guard)

            # ── Size callout box to its text ──────────────────────────────────
            lines      = text.split("\n")
            max_line_w = max((fitz.get_text_length(line, fontname="helv", fontsize=CALLOUT_FONT_SIZE) for line in lines), default=10.0)
            box_w      = max_line_w + 16.0  # 8pt padding on each side for safety
            box_h      = len(lines) * (CALLOUT_FONT_SIZE * 1.2) + 12.0  # 1.2 line height + 6pt top/bottom padding
            bw_img = box_w * (dpi / 72.0)
            bh_img = box_h * (dpi / 72.0)

            is_design_note = c.get("is_design_note", False)
            if is_design_note:
                # ── Find cleanest corner for Design Note ──────────────────────
                candidates = [
                    # Top-Left corner
                    (page_rect.x0 + 50.0, page_rect.y0 + 80.0),
                    # Bottom-Left corner
                    (page_rect.x0 + 50.0, page_rect.y1 - 80.0 - box_h),
                    # Bottom-Right corner
                    (page_rect.x1 - 50.0 - box_w, page_rect.y1 - 80.0 - box_h)
                ]
                
                best_corner_x, best_corner_y = None, None
                best_corner_density = 999.0
                
                for px_pdf, py_pdf in candidates:
                    px_img = (px_pdf + box_w / 2.0) * (dpi / 72.0)
                    py_img = (py_pdf + box_h / 2.0) * (dpi / 72.0)
                    
                    x1, y1 = int(px_img - bw_img / 2), int(py_img - bh_img / 2)
                    x2, y2 = int(px_img + bw_img / 2), int(py_img + bh_img / 2)
                    if x1 < 0 or y1 < 0 or x2 > w_img or y2 > h_img:
                        continue
                    
                    roi = img_gray[y1:y2, x1:x2]
                    if roi.size == 0:
                        continue
                    density = np.sum(roi < 240) / roi.size
                    
                    test_rect = fitz.Rect(px_pdf - 5, py_pdf - 5, px_pdf + box_w + 5, py_pdf + box_h + 5)
                    # Check overlap with existing placed_rects
                    if not any(test_rect.intersects(pr) for pr in placed_rects):
                        if density < best_corner_density:
                            best_corner_density = density
                            best_corner_x, best_corner_y = px_pdf, py_pdf
                
                if best_corner_x is not None:
                    text_rect = fitz.Rect(best_corner_x, best_corner_y, best_corner_x + box_w, best_corner_y + box_h)
                else:
                    # Fallback if all corners are full
                    text_rect = fitz.Rect(page_rect.x0 + 50.0, page_rect.y1 - 100.0 - box_h, page_rect.x0 + 50.0 + box_w, page_rect.y1 - 100.0)
                
                placed_rects.append(text_rect)
                hard_rects.append(text_rect)
                
                try:
                    annot = page.add_freetext_annot(
                        text_rect, c["text"], fontsize=CALLOUT_FONT_SIZE, fontname="helv",
                        text_color=(0, 0, 0), fill_color=(1, 1, 0),
                        align=1,
                    )
                    try:
                        annot.set_border(width=2.5)
                    except Exception:
                        pass
                    annot.update()
                    _patch_annot_color(doc, annot, font_size=CALLOUT_FONT_SIZE)
                except Exception as err:
                    logger.warning(f"Could not place design note: {err}")
                continue

            found = False
            best_px, best_py = None, None
            best_density = 999.0
            
            # Radii to search: starting close (80pt) to far (450pt)
            radii = np.arange(80 * (dpi / 72.0), 450 * (dpi / 72.0), 30 * (dpi / 72.0))
            angles = np.linspace(0, 2 * math.pi, 24, endpoint=False)
            
            # --- Attempt 1: Strict (clean background, no symbol guard overlap)
            for r_img in radii:
                for ang in angles:
                    px = cx_img + r_img * math.cos(ang)
                    py = cy_img + r_img * math.sin(ang)
                    
                    # Check image bounds
                    x1, y1 = int(px - bw_img / 2), int(py - bh_img / 2)
                    x2, y2 = int(px + bw_img / 2), int(py + bh_img / 2)
                    if x1 < 0 or y1 < 0 or x2 > w_img or y2 > h_img:
                        continue
                    
                    roi = img_gray[y1:y2, x1:x2]
                    if roi.size == 0:
                        continue
                    density = np.sum(roi < 240) / roi.size
                    
                    test_rect = fitz.Rect(
                        px * (72 / dpi) - box_w / 2 - 5, py * (72 / dpi) - box_h / 2 - 5,
                        px * (72 / dpi) + box_w / 2 + 5, py * (72 / dpi) + box_h / 2 + 5,
                    )
                    
                    if density < 0.08:
                        if not any(test_rect.intersects(pr) for pr in placed_rects):
                            best_px, best_py = px, py
                            found = True
                            break
                if found:
                    break
            
            # --- Attempt 2: Medium (allow slightly noisier background, no symbol guard overlap)
            if not found:
                for r_img in radii:
                    for ang in angles:
                        px = cx_img + r_img * math.cos(ang)
                        py = cy_img + r_img * math.sin(ang)
                        
                        x1, y1 = int(px - bw_img / 2), int(py - bh_img / 2)
                        x2, y2 = int(px + bw_img / 2), int(py + bh_img / 2)
                        if x1 < 0 or y1 < 0 or x2 > w_img or y2 > h_img:
                            continue
                        
                        roi = img_gray[y1:y2, x1:x2]
                        if roi.size == 0:
                            continue
                        density = np.sum(roi < 240) / roi.size
                        
                        test_rect = fitz.Rect(
                            px * (72 / dpi) - box_w / 2 - 5, py * (72 / dpi) - box_h / 2 - 5,
                            px * (72 / dpi) + box_w / 2 + 5, py * (72 / dpi) + box_h / 2 + 5,
                        )
                        
                        if density < 0.15:
                            if not any(test_rect.intersects(pr) for pr in placed_rects):
                                best_px, best_py = px, py
                                found = True
                                break
                    if found:
                        break
            
            # --- Attempt 3: Relaxed (allow symbol guard intersection, but no text box overlaps, clear background)
            if not found:
                for r_img in radii:
                    for ang in angles:
                        px = cx_img + r_img * math.cos(ang)
                        py = cy_img + r_img * math.sin(ang)
                        
                        x1, y1 = int(px - bw_img / 2), int(py - bh_img / 2)
                        x2, y2 = int(px + bw_img / 2), int(py + bh_img / 2)
                        if x1 < 0 or y1 < 0 or x2 > w_img or y2 > h_img:
                            continue
                        
                        roi = img_gray[y1:y2, x1:x2]
                        if roi.size == 0:
                            continue
                        density = np.sum(roi < 240) / roi.size
                        
                        test_rect = fitz.Rect(
                            px * (72 / dpi) - box_w / 2 - 5, py * (72 / dpi) - box_h / 2 - 5,
                            px * (72 / dpi) + box_w / 2 + 5, py * (72 / dpi) + box_h / 2 + 5,
                        )
                        
                        # Check against hard rects (other callout texts, title box)
                        if not any(test_rect.intersects(hr) for hr in hard_rects):
                            # Track best density position that avoids hard overlaps
                            if density < best_density:
                                best_density = density
                                best_px, best_py = px, py
                                if density < 0.10: # If it's a relatively clean background, accept it immediately
                                    found = True
                                    break
                    if found:
                        break

            # If we found a good relaxed candidate with low/moderate density, use it!
            if not found and best_px is not None and best_density < 0.30:
                found = True
            
            # --- Attempt 4: Fallback (absolute backup offset, ensuring we at least don't overlap hard rects)
            if not found:
                # If no clear space, default to 180px offset but try to find one that doesn't intersect hard_rects
                fallback_offsets = [
                    (180, -180), (-180, -180), (180, 180), (-180, 180),
                    (250, -250), (-250, -250), (250, 250), (-250, 250)
                ]
                for ox, oy in fallback_offsets:
                    px = cx_img + ox * (dpi / 72.0)
                    py = cy_img + oy * (dpi / 72.0)
                    test_rect = fitz.Rect(
                        px * (72 / dpi) - box_w / 2 - 5, py * (72 / dpi) - box_h / 2 - 5,
                        px * (72 / dpi) + box_w / 2 + 5, py * (72 / dpi) + box_h / 2 + 5,
                    )
                    if not any(test_rect.intersects(hr) for hr in hard_rects):
                        best_px, best_py = px, py
                        found = True
                        break
                
                if not found:
                    # absolute fallback if all offsets are blocked
                    best_px, best_py = cx_img + 200 * (dpi / 72.0), cy_img - 200 * (dpi / 72.0)
            
            ex_img, ey_img = best_px, best_py

            ex_pdf, ey_pdf = ex_img * (72 / dpi), ey_img * (72 / dpi)

            text_rect = fitz.Rect(
                ex_pdf - box_w / 2, ey_pdf - box_h / 2,
                ex_pdf + box_w / 2, ey_pdf + box_h / 2
            )
            placed_rects.append(text_rect)
            hard_rects.append(text_rect)

            # ── Arrow geometry (conditional straight line vs knee bend) ───────
            angle = math.atan2(cy_pdf - ey_pdf, cx_pdf - ex_pdf)
            offset_pts = ARROW_OFFSET_PTS  # Dynamically set based on DPI to prevent overlap
            tip_x = cx_pdf - offset_pts * math.cos(angle)
            tip_y = cy_pdf - offset_pts * math.sin(angle)

            # Structured attachment and horizontal knee shoulder
            if ex_pdf > cx_pdf:
                attach_x = ex_pdf - box_w / 2.0
                attach_y = ey_pdf
                knee_x = attach_x - 12.0 # 12pt horizontal shoulder
                knee_y = attach_y
            else:
                attach_x = ex_pdf + box_w / 2.0
                attach_y = ey_pdf
                knee_x = attach_x + 12.0 # 12pt horizontal shoulder
                knee_y = attach_y

            # Check if straight line from tip to attach crosses any dark pixels (writing/symbols)
            line_blocked = False
            pts_x = np.linspace(tip_x * (dpi / 72.0), attach_x * (dpi / 72.0), 12)
            pts_y = np.linspace(tip_y * (dpi / 72.0), attach_y * (dpi / 72.0), 12)
            for lpx, lpy in zip(pts_x[1:-1], pts_y[1:-1]):
                ilx, ily = int(lpx), int(lpy)
                if 0 <= ilx < w_img and 0 <= ily < h_img:
                    if img_gray[ily, ilx] < 210:
                        line_blocked = True
                        break

            is_vertically_aligned = abs(cy_pdf - ey_pdf) < 20.0
            is_horizontally_aligned = abs(cx_pdf - ex_pdf) < 20.0

            if not line_blocked and (is_vertically_aligned or is_horizontally_aligned):
                # Straight 2-point arrow when directly aligned and clear path
                callout_pts = [fitz.Point(tip_x, tip_y), fitz.Point(attach_x, attach_y)]
            elif not line_blocked and abs(cy_pdf - ey_pdf) < 35.0:
                # Straight 2-point arrow when small vertical delta and clear path
                callout_pts = [fitz.Point(tip_x, tip_y), fitz.Point(attach_x, attach_y)]
            else:
                # 3-point knee bend when significant offset or routing around writing/symbols
                callout_pts = [fitz.Point(tip_x, tip_y), fitz.Point(knee_x, knee_y), fitz.Point(attach_x, attach_y)]

            # ── Draw annotation ───────────────────────────────────────────────
            try:
                annot = page.add_freetext_annot(
                    text_rect, c["text"], fontsize=CALLOUT_FONT_SIZE, fontname="helv",
                    text_color=(0, 0, 0), fill_color=(1, 1, 0),
                    callout=callout_pts,
                    align=1,
                )

                try:
                    annot.set_border(width=1.0)
                except Exception:
                    pass
                annot.update()
                _patch_annot_color(doc, annot, font_size=CALLOUT_FONT_SIZE, is_callout=True)
            except Exception as err:
                logger.warning(f"Could not place callout '{text}': {err}")

    # ── Stamp survey image + title box ────────────────────────────────────────
    total_pages = doc.page_count
    for i in range(total_pages):
        _draw_legend_stack(
            doc[i], survey_image_path if i == 0 else None, title_box_data,
            title_font_size=title_font_size,
            page_num=i + 1,
            total_pages=total_pages,
            include_legend=include_legend and (i == 0),
            callouts=callout_records
        )

    # ── Final appearance stream patch pass across all pages and annotations ──
    for page_idx in range(doc.page_count):
        p = doc[page_idx]
        for a in list(p.annots()):
            if a.type[0] == 2:  # FreeText / FreeTextCallout
                xref = a.xref
                _, it_val = doc.xref_get_key(xref, "IT")
                is_callout = (it_val == "/FreeTextCallout")
                f_size = CALLOUT_FONT_SIZE if is_callout else title_font_size
                border_w = "1" if is_callout else "2.5"
                # Direct low-level xref dictionary modifications to preserve custom AP stream
                doc.xref_set_key(xref, "F", "4")  # fitz.PDF_ANNOT_IS_PRINT
                doc.xref_set_key(xref, "BS", f"<< /W {border_w} /S /S >>")  # 1pt for callouts, 2.5pt for title box
                doc.xref_set_key(xref, "C", "[1 0 0]")  # Stroke: Red
                doc.xref_set_key(xref, "IC", "[1 1 0]")  # Interior/Fill: Yellow
                doc.xref_set_key(xref, "DA", f"(0 0 0 rg /Helv {f_size} Tf)")
                doc.xref_set_key(
                    xref, "DS",
                    f"(font: Helv {f_size}pt; color: #000000; background-color: #FFFF00; border: {border_w}pt solid #FF0000;)"
                )
                ap_type, ap_val = doc.xref_get_key(xref, "AP")
                if ap_val and "/N" in ap_val:
                    m = re.search(r"/N\s+(\d+)", ap_val)
                    if m:
                        n_xref = int(m.group(1))
                        raw = doc.xref_stream(n_xref)
                        # 1. Replace RGB stroke with red RGB stroke (1 0 0 RG)
                        patched = re.sub(rb"\d+\.?\d*\s+\d+\.?\d*\s+\d+\.?\d*\s+RG", b"1 0 0 RG", raw)
                        # 2. Replace Grayscale stroke (e.g. 0 G) with red RGB stroke (1 0 0 RG)
                        patched = re.sub(rb"(?<!\w)\d+\.?\d*\s+G\b", b"1 0 0 RG", patched)
                        # 3. Stroke width: 1pt for callouts, 2.5pt for title box
                        if is_callout:
                            patched = re.sub(rb"\b2\.5\s+w\b", b"1 w", patched)
                            patched = re.sub(rb"\b1\.5\s+w\b", b"1 w", patched)
                            patched = re.sub(rb"\b0\s+w\b", b"1 w", patched)
                        else:
                            patched = re.sub(rb"\b0\s+w\b", b"2.5 w", patched)
                            patched = re.sub(rb"\b1\.5\s+w\b", b"2.5 w", patched)
                            patched = re.sub(rb"\b1\s+w\b", b"2.5 w", patched)
                        # 4. If fill-only rect without stroke exists, convert 're \n f' to 're \n B' (fill & stroke)
                        if b" re\nS" not in patched and b" re\r\nS" not in patched and b" re S" not in patched and b" re\nf\n" in patched:
                            patched = re.sub(rb"(\d+\.?\d*\s+\d+\.?\d*\s+\d+\.?\d*\s+\d+\.?\d*\s+re\s*\r?\n\s*)f\b", rb"\g<1>B", patched)
                        doc.update_stream(n_xref, patched)

    doc.save(str(output_path), deflate=True, clean=False)
    doc.close()
    logger.info(f"Report saved → {output_path}")
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# Public: Fiber Overview shortcut (no callout records, just stamp)
# ─────────────────────────────────────────────────────────────────────────────

def generate_final_report(
    pdf_path,
    callouts,           # may be a non-empty list from fiber overview
    output_path,
    dpi: int = 300,
    survey_image_path=None,
    title_box_data=None,
    title_font_size: int = 14,
):
    """
    Used by fiber overview pipeline.
    Passes callouts through to generate_vector_report so node/splice annotations
    are rendered on the map.
    """
    return generate_vector_report(
        after_pdf_path=pdf_path,
        callout_records=callouts if callouts else [],
        tile_offsets={},
        W_inv=np.eye(3),
        output_path=output_path,
        dpi=dpi,
        survey_image_path=survey_image_path,
        title_box_data=title_box_data,
        title_font_size=title_font_size,
        include_legend=False,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Private: Survey image + title box stamper
# ─────────────────────────────────────────────────────────────────────────────

def _draw_legend_stack(
    page: fitz.Page,
    survey_image_path,
    title_box_data: dict | None,
    title_font_size: int = 24,
    margin_pts: float = 15.0,
    include_legend: bool = False,
    page_num: int = 1,
    total_pages: int = 1,
    **kwargs,
):
    """
    Stamps the survey screenshot and Prism info title box.
    Placement rules:
      - Evaluates all 4 corners (TL, TR, BL, BR) to find the corner with the most white space.
      - If screenshot is in top corner (TL or TR), text box is placed directly below it.
      - If screenshot is in bottom corner (BL or BR), text box is placed next to it side-by-side.
      - Screenshot is stamped as an interactive, movable Stamp annotation.
      - Title box is stamped as an interactive, movable FreeText annotation.
      - For remaining pages (page_num > 1), ONLY the text box appears in the top corner (TR).
    """
    page_r = page.rect
    pw, ph = page_r.width, page_r.height
    gap = 8.0
    margin = margin_pts

    # ── Extract title fields ───────────────────────────────────────────────────
    pid      = ""
    node     = ""
    inst     = ""
    map_type = ""
    if title_box_data:
        pid      = title_box_data.get("prism_id", "")
        node     = title_box_data.get("node_name", "")
        inst     = title_box_data.get("instance", "")
        map_type = title_box_data.get("map_type", "")

    print_name = ""
    if map_type:
        m = map_type.upper()
        if "AFTER" in m:
            print_name = "AFTER"
        elif "BEFORE" in m:
            print_name = "BEFORE"
        elif "SCHEMATIC" in m:
            print_name = "SCHEMATIC REPORT"
        elif "OVERVIEW" in m:
            print_name = "OVERVIEW PRINT"
        else:
            print_name = m

    raw_lines = [
        pid if pid else "",
        node if node else "",
        inst if inst else "",
        print_name,
        f"PG {page_num} OF {total_pages}",
    ]
    lines = [l for l in raw_lines if l]

    # ── Measure text dimensions to scale image properly ────────────────────────
    font_size   = title_font_size
    line_h      = font_size * 1.5

    if "OVERVIEW" in map_type.upper():
        pad_w, pad_h = font_size * 0.8, font_size * 0.8
    else:
        pad_w, pad_h = font_size * 2.0, font_size * 2.0
    max_line_w  = 0
    if lines:
        max_line_w = max(fitz.get_text_length(l, fontname="helv", fontsize=font_size) for l in lines)

    box_w = max_line_w + pad_w
    box_h = line_h * len(lines) + pad_h if title_box_data else 0

    # ── Survey screenshot dimensions ───────────────────────────────────────────
    ss_w = 0.0
    ss_h = 0.0
    has_screenshot = bool(survey_image_path and Path(survey_image_path).exists() and page_num == 1)
    if has_screenshot:
        if "SCHEMATIC" in map_type.upper():
            target_ss_w = max(1200.0, box_w * 2.0)
        elif "OVERVIEW" in map_type.upper():
            target_ss_w = max(250.0, box_w * 0.8)
        else:
            target_ss_w = max(800.0, box_w * 1.5)
        try:
            tmp = fitz.open(str(survey_image_path))
            ir = tmp[0].rect
            tmp.close()
            ss_w = min(target_ss_w, pw * 0.45)
            ss_h = (ir.height / ir.width) * ss_w
        except Exception:
            ss_w = min(target_ss_w, pw * 0.45)
            ss_h = ss_w * 0.66

    # ── Legend Size ────────────────────────────────────────────────────────────
    if "COAX" in map_type.upper() and ss_w > 0:
        leg_scale = (ss_w / 7.5) / 260.0
    else:
        leg_scale = title_font_size / 14.0

    leg_line_h = 18 * leg_scale
    leg_col_w = [35 * leg_scale, 120 * leg_scale, 35 * leg_scale]
    leg_w = sum(leg_col_w)
    leg_h = leg_line_h * (len(LEGEND_DATA) + 1) if include_legend else 0

    # ── Determine Placement Coordinates ────────────────────────────────────────
    if not has_screenshot or ss_h == 0:
        # Remaining pages (page_num > 1) or no screenshot: Top corner (TR) text box only
        box_rect = fitz.Rect(pw - box_w - margin, margin, pw - margin, margin + box_h)
        ss_rect = None
        leg_rect = fitz.Rect(pw - leg_w - margin, margin + box_h + gap, pw - margin, margin + box_h + gap + leg_h) if include_legend else None
    else:
        # Page 1 with screenshot: evaluate all 4 corners (TL, TR, BL, BR)
        candidates = {
            "TL": {
                "ss": fitz.Rect(margin, margin, margin + ss_w, margin + ss_h),
                "box": fitz.Rect(margin, margin + ss_h + gap, margin + box_w, margin + ss_h + gap + box_h),
                "leg": fitz.Rect(margin, margin + ss_h + gap + box_h + gap, margin + leg_w, margin + ss_h + gap + box_h + gap + leg_h) if include_legend else None,
                "roi": fitz.Rect(margin, margin, margin + max(ss_w, box_w, leg_w if include_legend else 0), margin + ss_h + gap + box_h + (gap + leg_h if include_legend else 0)),
            },
            "TR": {
                "ss": fitz.Rect(pw - ss_w - margin, margin, pw - margin, margin + ss_h),
                "box": fitz.Rect(pw - box_w - margin, margin + ss_h + gap, pw - margin, margin + ss_h + gap + box_h),
                "leg": fitz.Rect(pw - leg_w - margin, margin + ss_h + gap + box_h + gap, pw - margin, margin + ss_h + gap + box_h + gap + leg_h) if include_legend else None,
                "roi": fitz.Rect(pw - max(ss_w, box_w, leg_w if include_legend else 0) - margin, margin, pw - margin, margin + ss_h + gap + box_h + (gap + leg_h if include_legend else 0)),
            },
            "BL": {
                "ss": fitz.Rect(margin, ph - margin - ss_h, margin + ss_w, ph - margin),
                "box": fitz.Rect(margin + ss_w + gap, ph - margin - box_h, margin + ss_w + gap + box_w, ph - margin),
                "leg": fitz.Rect(margin + ss_w + gap, ph - margin - box_h - gap - leg_h, margin + ss_w + gap + leg_w, ph - margin - box_h - gap) if include_legend else None,
                "roi": fitz.Rect(margin, ph - margin - max(ss_h, box_h + (gap + leg_h if include_legend else 0)), margin + ss_w + gap + max(box_w, leg_w if include_legend else 0), ph - margin),
            },
            "BR": {
                "ss": fitz.Rect(pw - margin - ss_w, ph - margin - ss_h, pw - margin, ph - margin),
                "box": fitz.Rect(pw - margin - ss_w - gap - box_w, ph - margin - box_h, pw - margin - ss_w - gap, ph - margin),
                "leg": fitz.Rect(pw - margin - ss_w - gap - max(box_w, leg_w), ph - margin - box_h - gap - leg_h, pw - margin - ss_w - gap, ph - margin - box_h - gap) if include_legend else None,
                "roi": fitz.Rect(pw - margin - ss_w - gap - max(box_w, leg_w if include_legend else 0), ph - margin - max(ss_h, box_h + (gap + leg_h if include_legend else 0)), pw - margin, ph - margin),
            },
        }

        # Analyze raster brightness in each corner
        best_corner = "TR"
        try:
            pix = page.get_pixmap(dpi=100)
            if pix.n == 4:
                gray = cv2.cvtColor(np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 4), cv2.COLOR_RGBA2GRAY)
            else:
                gray = cv2.cvtColor(np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n), cv2.COLOR_RGB2GRAY)
            scale = 100.0 / 72.0

            scores = {}
            for name, cfg in candidates.items():
                r = cfg["roi"]
                x0 = max(0, int(r.x0 * scale))
                y0 = max(0, int(r.y0 * scale))
                x1 = min(gray.shape[1], int(r.x1 * scale))
                y1 = min(gray.shape[0], int(r.y1 * scale))
                roi = gray[y0:y1, x0:x1]
                if roi.size == 0:
                    scores[name] = 0.0
                else:
                    w_ratio = float(np.mean(roi >= 250))
                    m_val = float(np.mean(roi))
                    scores[name] = w_ratio * 1000.0 + m_val

            # If TR has plenty of white space, prefer TR
            if scores.get("TR", 0) >= 980.0:
                best_corner = "TR"
            else:
                best_corner = max(scores, key=lambda k: scores[k])
            logger.info(f"Page {page_num} chosen corner: {best_corner} (scores: {scores})")
        except Exception as e:
            logger.warning(f"Whitest corner detection fallback to TR: {e}")
            best_corner = "TR"

        ss_rect = candidates[best_corner]["ss"]
        box_rect = candidates[best_corner]["box"]
        leg_rect = candidates[best_corner]["leg"]

    # ── 1. Draw Movable Survey Screenshot (Stamp Annotation) ──────────────────
    if ss_rect and survey_image_path:
        _stamp_movable_image(page, ss_rect, survey_image_path)

    # ── 2. Draw Movable Title Box (FreeText Annotation) ───────────────────────
    if title_box_data and box_rect:
        box_text = "\n".join(lines)
        try:
            annot = page.add_freetext_annot(
                box_rect,
                box_text,
                fontsize=font_size,
                fontname="helv",
                text_color=(0, 0, 0),
                fill_color=(1, 1, 0),
                align=fitz.TEXT_ALIGN_CENTER,
            )
            try:
                annot.set_border(width=2.5, dashes=None)
            except Exception:
                pass
            annot.update()
            _patch_annot_color(page.parent, annot, font_size=font_size, is_callout=False)
        except Exception as err:
            logger.warning(f"Could not place title box: {err}")

    # ── 3. Draw Legend Table if requested ──────────────────────────────────────
    if include_legend and leg_rect:
        leg_x0 = leg_rect.x0
        curr_y = leg_rect.y0
        page.draw_rect(leg_rect, color=(1, 0, 0), fill=(1, 1, 1), width=2.5)

        counts = {k: 0 for k in LEGEND_DATA}
        callouts = kwargs.get("callouts", [])
        for c in callouts:
            t = c.get("text", "")
            if t:
                first = t[0].upper()
                if first in counts:
                    counts[first] += 1

        ry = curr_y
        hdr_font = max(8, int(11 * leg_scale))
        val_font = max(8, int(10 * leg_scale))

        for i, hdr in enumerate(["Code", "Action", "Count"]):
            xc = leg_x0 + sum(leg_col_w[:i])
            cr = fitz.Rect(xc, ry, xc + leg_col_w[i], ry + leg_line_h)
            page.draw_rect(cr, color=(0, 0, 0), fill=(0.8, 0.9, 1.0), width=1.0)
            tw_pts = fitz.get_text_length(hdr, fontname="hebo", fontsize=hdr_font)
            page.insert_text((xc + (leg_col_w[i] - tw_pts) / 2, ry + (leg_line_h + hdr_font * 0.8) / 2), hdr, fontsize=hdr_font, fontname="hebo")
        ry += leg_line_h

        for idx, code in enumerate(sorted(LEGEND_DATA)):
            for i, val in enumerate([code, LEGEND_DATA[code], str(counts[code])]):
                xc = leg_x0 + sum(leg_col_w[:i])
                cr = fitz.Rect(xc, ry, xc + leg_col_w[i], ry + leg_line_h)
                page.draw_rect(cr, color=(0, 0, 0), width=1.0)
                tw_pts = fitz.get_text_length(val, fontname="helv", fontsize=val_font)
                page.insert_text((xc + (leg_col_w[i] - tw_pts) / 2, ry + (leg_line_h + val_font * 0.8) / 2), val, fontsize=val_font, fontname="helv")
            ry += leg_line_h
