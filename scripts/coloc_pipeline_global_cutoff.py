#!/usr/bin/env python3
# coloc_pipeline_ch1global.py
#
# - CH1 uses GLOBAL cutoffs learned from negatives (restricted to paths containing "control")
# - CH2 uses LOCAL thresholding per image/compartment (Otsu by default, or percentile)
# - Background subtraction (morph. opening)
# - DAPI-based nucleus/cytoplasm masks
# - Intensity-weighted colocalization (IWC) computed from overlap mask
# - Optional Pearson
# - Optional join with cellpose counts CSV (by MAX_NUMBER)
# - QA overlays
#
# Example:
#   python coloc_pipeline_ch1global.py \
#     --root_dir Batch_1 \
#     --out_csv colocalization_results.csv \
#     --qa_dir qa_out \
#     --neg_groups "control: no hpg" "chx + tig" \
#     --neg_ch1_percentile 99.5 \
#     --require_control_in_path \
#     --ch2_method otsu \
#     --exclude composite
#
import os
import re
import json
import argparse
import logging
import numpy as np
import pandas as pd
import tifffile as tiff
import matplotlib.pyplot as plt

from skimage import filters, morphology
from skimage.morphology import disk, opening
from scipy.stats import pearsonr

logging.basicConfig(
    filename="coloc_pipeline_ch1global.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# ------------------------- utilities -------------------------

def list_image_files(root_dir, include=None, exclude=None, exts=(".tif", ".tiff")):
    files = []
    inc = [s.lower() for s in (include or [])]
    exc = [s.lower() for s in (exclude or [])]
    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            if not fn.lower().endswith(exts):
                continue
            path = os.path.join(dirpath, fn)
            low = path.lower()
            if inc and not any(s in low for s in inc):
                continue
            if exc and any(s in low for s in exc):
                continue
            files.append(path)
    files.sort()
    return files

def get_max_number(text):
    m = re.search(r"(MAX_\d+)", os.path.basename(text))
    return m.group(1) if m else "UNKNOWN"

def path_has_control(fp):
    return "control" in os.path.dirname(fp).lower()

def subtract_background(image, selem_radius=50):
    selem = disk(selem_radius)
    background = opening(image, selem)
    corrected = image.astype(np.float32) - background.astype(np.float32)
    corrected[corrected < 0] = 0
    return corrected

def make_tissue_mask(ch1, ch2, dapi):
    # Simple tissue mask from summed channels; then clean
    summ = ch1 + ch2 + dapi.astype(np.float32)
    thr = filters.threshold_otsu(summ)
    mask = summ > thr
    mask = morphology.remove_small_objects(mask, min_size=200)
    mask = morphology.remove_small_holes(mask, area_threshold=200)
    mask = morphology.binary_dilation(mask, morphology.disk(2))
    return mask

def create_nucleus_and_cyto_masks(dapi_channel, tissue_mask=None):
    thr = filters.threshold_otsu(dapi_channel)
    nucleus = dapi_channel > thr
    nucleus = morphology.remove_small_objects(nucleus, min_size=50)
    nucleus = morphology.remove_small_holes(nucleus, area_threshold=50)

    if tissue_mask is None:
        tissue_mask = morphology.binary_dilation(nucleus, morphology.disk(3))

    cyto = tissue_mask & (~nucleus)
    cyto = morphology.remove_small_objects(cyto, min_size=50)
    cyto = morphology.remove_small_holes(cyto, area_threshold=50)
    return nucleus, cyto

def collapse_condition(filename):
    f = filename.lower()
    if "hpg only" in f: return "control: hpg only"
    if "no hpg"  in f: return "control: no hpg"

    if "chx" in f and all(x not in f for x in ["crb","tig","oxa","veh","nc"]): return "chx only"
    if "tig" in f and all(x not in f for x in ["chx","crb","veh","oxa","nc"]): return "tig only"
    if "crb" in f and all(x not in f for x in ["chx","tig","veh","oxa","nc"]): return "crb only"
    if "oxa" in f and "chx" not in f and "veh" not in f and "crb" not in f:  return "oxa only"

    if "chx" in f and "veh" in f: return "chx + veh"
    if "chx" in f and "crb" in f: return "chx + crb"
    if "chx" in f and "tig" in f: return "chx + tig"
    if "tig" in f and "crb" in f and "chx" not in f: return "tig + crb"
    if "oxa" in f and "veh" in f: return "oxa + veh"
    if "oxa" in f and "crb" in f: return "oxa + crb"
    if "nc"  in f and "veh" in f: return "nc + veh"
    if "nc"  in f and "crb" in f: return "nc + crb"
    if "nc"  in f and all(x not in f for x in ["chx","oxa","crb","veh","tig"]): return "siRNA nc"
    return "unspecified"

def compute_pearson_correlation(ch1, ch2):
    a = ch1.ravel().astype(np.float32)
    b = ch2.ravel().astype(np.float32)
    if a.size < 2 or b.size < 2:
        return np.nan
    try:
        return float(pearsonr(a, b)[0])
    except Exception:
        return np.nan

# ------------------------- GLOBAL CUTOFF (CH1 ONLY) -------------------------

def compute_global_ch1_cutoffs(
    file_paths,
    selem_radius=50,
    dapi_index=2,
    neg_groups=("control: no hpg", "chx + tig"),
    ch1_percentile=99.5,
    require_control_in_path=True,
    save_json_path=None,
):
    """
    Learn absolute CH1 intensity cutoffs from NEGATIVE references only.
    Reference images MUST live in a directory whose path contains 'control' (if require_control_in_path).
    Returns per-compartment thresholds for CH1: ch1_thresh_nuc, ch1_thresh_cyto
    """
    ch1_nuc_vals, ch1_cyt_vals = [], []

    for fp in file_paths:
        try:
            if require_control_in_path and not path_has_control(fp):
                continue

            fname = os.path.basename(fp)
            cond = collapse_condition(fname)
            if cond not in neg_groups:
                continue

            img = tiff.imread(fp)
            if img.ndim < 3:
                logging.warning(f"Skipping (not C,H,W): {fp}")
                continue
            ch1_raw, ch2_raw, dapi = img[0], img[1], img[dapi_index]
            ch1 = subtract_background(ch1_raw, selem_radius=selem_radius)
            ch2 = subtract_background(ch2_raw, selem_radius=selem_radius)  # just for tissue mask

            tissue = make_tissue_mask(ch1, ch2, dapi)
            nuc, cyto = create_nucleus_and_cyto_masks(dapi, tissue)

            if nuc.sum() > 0:
                ch1_nuc_vals.append(ch1[nuc])
            if cyto.sum() > 0:
                ch1_cyt_vals.append(ch1[cyto])

        except Exception as e:
            logging.error(f"CH1 cutoff scan failed for {fp}: {e}")

    def concat(arrs):
        return np.concatenate([a.ravel() for a in arrs]) if arrs else np.array([])

    ch1_n = concat(ch1_nuc_vals)
    ch1_c = concat(ch1_cyt_vals)

    def perc(v, p):
        return float(np.percentile(v, p)) if v.size else np.nan

    cuts = {
        "ch1_thresh_nuc": perc(ch1_n, ch1_percentile),
        "ch1_thresh_cyto": perc(ch1_c, ch1_percentile),
        "params": {
            "neg_groups": list(neg_groups),
            "require_control_in_path": bool(require_control_in_path),
            "ch1_percentile": float(ch1_percentile),
        }
    }
    logging.info(f"Global CH1 cutoffs: {cuts}")
    if save_json_path:
        try:
            with open(save_json_path, "w") as f:
                json.dump(cuts, f, indent=2)
        except Exception as e:
            logging.error(f"Could not save CH1 cutoffs JSON: {e}")
    return cuts

def positive_mask_global(image, mask, abs_threshold):
    if mask.sum() == 0 or abs_threshold is None or (isinstance(abs_threshold, float) and np.isnan(abs_threshold)):
        return np.zeros_like(mask, dtype=bool)
    out = np.zeros_like(mask, dtype=bool)
    out[mask] = image[mask] >= abs_threshold
    return out

# ------------------------- CH2 LOCAL THRESHOLDING -------------------------

def positive_mask_local(image, mask, method="otsu", percentile=99.0):
    """
    Local (per-image, per-compartment) positives for CH2.
    method='otsu' (default) or 'percentile' (e.g., 99.0).
    """
    if mask.sum() == 0:
        return np.zeros_like(mask, dtype=bool)

    vals = image[mask]
    if vals.size == 0:
        return np.zeros_like(mask, dtype=bool)

    if method == "otsu":
        thr = filters.threshold_otsu(vals)
    elif method == "percentile":
        thr = np.percentile(vals, percentile)
    else:
        raise ValueError("method must be 'otsu' or 'percentile'")

    out = np.zeros_like(mask, dtype=bool)
    out[mask] = image[mask] >= thr
    return out

# ------------------------- METRICS & PLOTTING -------------------------

def intensity_weighted_coloc_from_mask(ch1, ch2, overlap_mask):
    if overlap_mask.sum() == 0:
        return np.nan, 0.0
    num = float(ch1[overlap_mask].sum() + ch2[overlap_mask].sum())
    den = float((ch1 + ch2).sum())
    return (100.0 * num / den) if den > 0 else np.nan, num

def plot_filtered_visualization(out_png, ch1_show, ch2_show, dapi_mask_rgb, ch1_pos_mask, ch2_pos_mask, ovl_mask):
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))

    def norm01(x):
        m = x.max()
        return x / (m + 1e-6)

    axes[0].imshow(norm01(ch1_show), cmap='Reds')
    axes[0].set_title("Ch1 (bg-sub)")

    axes[1].imshow(norm01(ch2_show), cmap='Greens')
    axes[1].set_title("Ch2 (bg-sub)")

    axes[2].imshow(dapi_mask_rgb, cmap='Blues')
    axes[2].set_title("DAPI (nucleus mask)")

    overlay = np.zeros((*ch1_pos_mask.shape, 3), dtype=np.float32)
    overlay[..., 0] = ch1_pos_mask.astype(float)  # red = CH1+
    overlay[..., 1] = ch2_pos_mask.astype(float)  # green = CH2+
    axes[3].imshow(overlay)
    axes[3].set_title("Positives (CH1=global, CH2=local)")

    axes[4].imshow(ovl_mask, cmap='gray')
    axes[4].set_title("Overlap (CH1+ & CH2+)")

    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

# ------------------------- MAIN -------------------------

def main():
    ap = argparse.ArgumentParser(description="Colocalization: CH1 global cutoff (from negatives), CH2 local thresholding.")
    ap.add_argument("--root_dir", required=True, help="Top-level directory to scan recursively for images")
    ap.add_argument("--out_csv", default="colocalization_results.csv", help="Output CSV path")
    ap.add_argument("--qa_dir", default="qa_out", help="Directory to save QA PNGs")
    ap.add_argument("--include", nargs="*", default=None, help="Only include paths containing any of these substrings")
    ap.add_argument("--exclude", nargs="*", default=None, help="Exclude paths containing any of these substrings")
    ap.add_argument("--dapi_index", type=int, default=2, help="Channel index for DAPI in (C,H,W) arrays")
    ap.add_argument("--selem_radius", type=int, default=50, help="Morph. opening radius for background subtraction")
    ap.add_argument("--compute_pearson", action="store_true", help="Also compute Pearson on bg-sub channels")
    # negatives for CH1
    ap.add_argument("--neg_groups", nargs="+",
                    default=["control: no hpg", "chx + tig"],
                    help="Condition labels considered NEGATIVE for learning CH1 global cutoff.")
    ap.add_argument("--neg_ch1_percentile", type=float, default=99.5,
                    help="Percentile for CH1 negative cutoff (e.g., 99, 99.5, 99.9).")
    ap.add_argument("--require_control_in_path", action="store_true", default=True,
                    help="Only learn cutoffs from files whose path contains 'control'.")
    ap.add_argument("--ch1_cutoffs_json", default=None, help="Optional path to save learned CH1 global cutoffs JSON.")
    # CH2 local thresholding
    ap.add_argument("--ch2_method", choices=["otsu", "percentile"], default="otsu",
                    help="Local thresholding method for CH2.")
    ap.add_argument("--ch2_percentile", type=float, default=99.0,
                    help="If --ch2_method percentile, use this percentile inside each compartment.")
    # Optional: cellpose counts CSV
    ap.add_argument("--cellpose_csv", default=None, help="Optional CSV with counts; must contain MAX_NUMBER and nuclei_sam")

    args = ap.parse_args()

    # List files
    file_paths = list_image_files(args.root_dir, include=args.include, exclude=args.exclude)
    if not file_paths:
        print("No images found. Check --root_dir and filters.")
        return

    print('============== Computing Global Channel 1 Thresholds =====================')
    # Learn CH1 global cutoffs (nucleus/cyto) from negatives
    ch1_cuts = compute_global_ch1_cutoffs(
        file_paths=file_paths,
        selem_radius=args.selem_radius,
        dapi_index=args.dapi_index,
        neg_groups=tuple(args.neg_groups),
        ch1_percentile=args.neg_ch1_percentile,
        require_control_in_path=args.require_control_in_path,
        save_json_path=args.ch1_cutoffs_json,
    )
    print("Global CH1 cutoffs:", json.dumps(ch1_cuts, indent=2, default=lambda x: None))

    # Optional: load cellpose counts
    counts = None
    if args.cellpose_csv and os.path.exists(args.cellpose_csv):
        try:
            cdf = pd.read_csv(args.cellpose_csv)
            cols = {c.lower(): c for c in cdf.columns}
            if "max_number" in cols and "nuclei_sam" in cols:
                counts = cdf.rename(columns={cols["max_number"]: "MAX_NUMBER", cols["nuclei_sam"]: "nuclei_sam"})
            else:
                rename_map = {}
                for c in cdf.columns:
                    lc = c.lower()
                    if lc == "max_number": rename_map[c] = "MAX_NUMBER"
                    if lc == "nuclei_sam": rename_map[c] = "nuclei_sam"
                counts = cdf.rename(columns=rename_map)
        except Exception as e:
            logging.error(f"Failed to load cellpose_csv: {e}")
            counts = None

    rows = []
    print('===============Processing Files=================')
    for fp in file_paths:
        try:
            img = tiff.imread(fp)
            if img.ndim < 3:
                logging.warning(f"Skipping (not C,H,W): {fp}")
                continue
            ch1_raw, ch2_raw, dapi = img[0], img[1], img[args.dapi_index] 

            # Background subtraction
            ch1 = subtract_background(ch1_raw, selem_radius=args.selem_radius)
            ch2 = subtract_background(ch2_raw, selem_radius=args.selem_radius)

            # Masks
            tissue = make_tissue_mask(ch1, ch2, dapi)
            nuc, cyto = create_nucleus_and_cyto_masks(dapi, tissue)

            # CH1 positives via GLOBAL absolute thresholds
            pos1_n = positive_mask_global(ch1, nuc,  ch1_cuts.get("ch1_thresh_nuc"))
            pos1_c = positive_mask_global(ch1, cyto, ch1_cuts.get("ch1_thresh_cyto"))

            # CH2 positives via LOCAL per-image thresholds
            pos2_n = positive_mask_local(ch2, nuc,  method=args.ch2_method, percentile=args.ch2_percentile)
            pos2_c = positive_mask_local(ch2, cyto, method=args.ch2_method, percentile=args.ch2_percentile)

            # Overlap masks
            ovl_n_mask = pos1_n & pos2_n
            ovl_c_mask = pos1_c & pos2_c

            # Metrics
            iwc_nuc, overlap_intensity_nuc = intensity_weighted_coloc_from_mask(ch1, ch2, ovl_n_mask)
            iwc_cyt, overlap_intensity_cyt = intensity_weighted_coloc_from_mask(ch1, ch2, ovl_c_mask)

            pearson_val = compute_pearson_correlation(ch1, ch2) if args.compute_pearson else np.nan

            # Row
            fname = os.path.basename(fp)
            session = get_max_number(fname)
            cond = collapse_condition(fname)

            row = {
                "MAX_NUMBER": session,
                "file": fname,
                "Path": fp,
                "Condition": cond,
                "IWC_Nucleus_%": iwc_nuc,
                "IWC_Cytoplasm_%": iwc_cyt,
                "OverlapIntensity_Nucleus": overlap_intensity_nuc,
                "OverlapIntensity_Cytoplasm": overlap_intensity_cyt,
                "Ch1_Thresh_Nuc_Global": ch1_cuts.get("ch1_thresh_nuc"),
                "Ch1_Thresh_Cyto_Global": ch1_cuts.get("ch1_thresh_cyto"),
                "Ch2_Method": args.ch2_method,
                "Ch2_Percentile": (args.ch2_percentile if args.ch2_method == "percentile" else np.nan),
                "Pixels_Nucleus": int(nuc.sum()),
                "Pixels_Cytoplasm": int(cyto.sum()),
                "OverlapPixels_Nucleus": int(ovl_n_mask.sum()),
                "OverlapPixels_Cytoplasm": int(ovl_c_mask.sum()),
                "Pearson_AllPixels": pearson_val,
            }

            # Optional join with counts
            if counts is not None and "MAX_NUMBER" in counts.columns and session in set(counts["MAX_NUMBER"].astype(str)):
                sub = counts[counts["MAX_NUMBER"].astype(str) == session]
                if not sub.empty and "nuclei_sam" in sub.columns:
                    row["nuclei_sam"] = float(sub["nuclei_sam"].iloc[0])

            rows.append(row)

            # QA plot
            dapi_mask_rgb = dapi * nuc.astype(dapi.dtype)
            out_png = os.path.join(args.qa_dir, session, os.path.splitext(fname)[0] + ".png")
            plot_filtered_visualization(
                out_png, ch1, ch2, dapi_mask_rgb,
                (pos1_n | pos1_c), (pos2_n | pos2_c), (ovl_n_mask | ovl_c_mask)
            )

            print(f"[OK] {fname}  IWC_nuc={iwc_nuc:.3f}  IWC_cyto={iwc_cyt:.3f}")

        except Exception as e:
            logging.error(f"Error processing {fp}: {e}")
            print(f"[ERR] {fp} :: {e}")

    df = pd.DataFrame(rows)
    order_cols = [
        "MAX_NUMBER", "file", "Path", "Condition",
        "IWC_Nucleus_%", "IWC_Cytoplasm_%",
        "OverlapIntensity_Nucleus", "OverlapIntensity_Cytoplasm",
        "Ch1_Thresh_Nuc_Global", "Ch1_Thresh_Cyto_Global",
        "Ch2_Method", "Ch2_Percentile",
        "Pixels_Nucleus", "Pixels_Cytoplasm",
        "OverlapPixels_Nucleus", "OverlapPixels_Cytoplasm",
        "Pearson_AllPixels", "nuclei_sam"
    ]
    for c in order_cols:
        if c not in df.columns:
            df[c] = np.nan
    df = df[order_cols]

    df.to_csv(args.out_csv, index=False)
    print(f"Saved results to {args.out_csv}")

if __name__ == "__main__":
    main()
