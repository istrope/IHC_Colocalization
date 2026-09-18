#!/usr/bin/env python3
# coloc_pipeline.py
import os, re, argparse, logging
import numpy as np
import pandas as pd
import tifffile as tiff
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.stats import pearsonr
from skimage import filters, morphology
from skimage.morphology import disk, opening, dilation

logging.basicConfig(
    filename="colocalization_analysis.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

def get_max_number(text):
    m = re.search(r"(MAX_\d+)", os.path.basename(text))
    return m.group(1) if m else "UNKNOWN"

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

def subtract_background(image, selem_radius=50):
    selem = disk(selem_radius)
    background = opening(image, selem)
    x = image.astype(np.float32) - background.astype(np.float32)
    x[x < 0] = 0
    return x

def make_tissue_mask(ch1, ch2, dapi):
    stack = ch1 + ch2 + dapi
    thr = filters.threshold_otsu(stack)
    tissue = stack > thr
    tissue = morphology.remove_small_objects(tissue, min_size=200)
    tissue = morphology.remove_small_holes(tissue, area_threshold=200)
    return tissue

def create_nucleus_and_cyto_masks(dapi_channel, tissue_mask=None, nuc_dilate=2, cyto_width=6):
    thr = filters.threshold_otsu(dapi_channel)
    nucleus = dapi_channel > thr
    nucleus = morphology.remove_small_objects(nucleus, min_size=50)
    nuc_expanded = dilation(nucleus, disk(nuc_dilate))
    cyto_ring = dilation(nuc_expanded, disk(cyto_width)) & (~nuc_expanded)
    if tissue_mask is not None:
        nucleus = nucleus & tissue_mask
        cyto_ring = cyto_ring & tissue_mask
    return nucleus, cyto_ring

def positive_mask(image, mask, method="otsu", percentile=80):
    vals = image[mask]
    out = np.zeros_like(image, dtype=bool)
    if vals.size == 0:
        return out
    if method == "otsu":
        thr = filters.threshold_otsu(vals)
    elif method == "percentile":
        thr = np.percentile(vals, percentile)
    else:
        raise ValueError("Unknown method for positive_mask")
    out[mask] = image[mask] > thr
    return out

def manders_coefficients(ch1, ch2, mask, pos1, pos2):
    I1 = ch1[mask]; I2 = ch2[mask]
    s1 = float(I1.sum()); s2 = float(I2.sum())
    if s1 <= 0 or s2 <= 0: return np.nan, np.nan
    M1 = float(ch1[pos2 & mask].sum()) / s1
    M2 = float(ch2[pos1 & mask].sum()) / s2
    return M1, M2

def dice_jaccard(bin1, bin2, mask):
    a = (bin1 & mask); b = (bin2 & mask)
    inter = int((a & b).sum()); s1 = int(a.sum()); s2 = int(b.sum())
    union = int((a | b).sum())
    dice = (2*inter)/(s1+s2) if (s1+s2)>0 else np.nan
    jacc = inter/union if union>0 else np.nan
    return dice, jacc

def intensity_weighted_coloc(ch1, ch2, mask, pos_mask=None):
    region = mask if pos_mask is None else (mask & pos_mask)
    if int(region.sum()) == 0: return np.nan, np.nan
    I1 = ch1[region]; I2 = ch2[region]
    num = float(np.minimum(I1, I2).sum())
    denom = float((I1 + I2).sum())
    iwc = 100.0 * num / denom if denom > 0 else np.nan
    return iwc, num

def pearson_on_region(ch1, ch2, region):
    if int(region.sum()) == 0: return np.nan
    return pearsonr(ch1[region].ravel(), ch2[region].ravel())[0]

def compute_control_baselines(file_paths, selem_radius=50):
    stats = {"nucleus": [], "cyto": []}
    for fp in file_paths:
        if "hpg only" not in os.path.basename(fp).lower(): 
            continue
        try:
            img = tiff.imread(fp)
            ch1_raw, ch2_raw, dapi = img[0], img[1], img[2]
            ch1 = subtract_background(ch1_raw, selem_radius=selem_radius)
            ch2 = subtract_background(ch2_raw, selem_radius=selem_radius)
            tissue = make_tissue_mask(ch1, ch2, dapi)
            nuc, cyto = create_nucleus_and_cyto_masks(dapi, tissue)
            pos1_n = positive_mask(ch1, nuc, method="otsu")
            pos2_n = positive_mask(ch2, nuc, method="otsu")
            pos1_c = positive_mask(ch1, cyto, method="otsu")
            pos2_c = positive_mask(ch2, cyto, method="otsu")
            region_n = nuc & (pos1_n | pos2_n)
            region_c = cyto & (pos1_c | pos2_c)
            def per_pixel_total(I1, I2, region):
                A = int(region.sum())
                return ((I1[region] + I2[region]).sum() / A) if A > 0 else np.nan
            mu_n = per_pixel_total(ch1, ch2, region_n)
            mu_c = per_pixel_total(ch1, ch2, region_c)
            if not np.isnan(mu_n): stats["nucleus"].append(mu_n)
            if not np.isnan(mu_c): stats["cyto"].append(mu_c)
        except Exception as e:
            logging.error(f"Control baseline failed for {fp}: {e}")
    baselines = {
        "nucleus": (np.nanmedian(stats["nucleus"]) if stats["nucleus"] else None),
        "cyto":    (np.nanmedian(stats["cyto"]) if stats["cyto"] else None),
    }
    logging.info(f"Control baselines (per-pixel): {baselines}")
    return baselines

def normalized_iwc_ratio(ch1, ch2, mask, baseline_per_pixel, pos_mask=None):
    region = mask if pos_mask is None else (mask & pos_mask)
    A = int(region.sum())
    if A == 0 or baseline_per_pixel is None or np.isnan(baseline_per_pixel):
        return np.nan
    num = float(np.minimum(ch1[region], ch2[region]).sum())
    per_pixel_overlap = num / A
    return per_pixel_overlap / (baseline_per_pixel + 1e-12)

def plot_filtered_visualization(out_png, ch1_show, ch2_show, dapi_masked, bin1, bin2, pos_hi_union=None):
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    axes[0].imshow(ch1_show, cmap='Reds');   axes[0].set_title("Ch1 (norm)")
    axes[1].imshow(ch2_show, cmap='Greens'); axes[1].set_title("Ch2 (norm)")
    axes[2].imshow(dapi_masked, cmap='Blues'); axes[2].set_title("DAPI × Nucleus")
    overlay = np.dstack((bin1.astype(float), bin2.astype(float), np.zeros_like(bin1, dtype=float)))
    axes[3].imshow(overlay); axes[3].set_title("Binary Positives")
    if pos_hi_union is None: pos_hi_union = bin1 & bin2
    hi = np.dstack((pos_hi_union.astype(float), pos_hi_union.astype(float), np.zeros_like(bin1, dtype=float)))
    axes[4].imshow(hi); axes[4].set_title("High-Intensity Union (proxy)")
    for ax in axes: ax.axis('off')
    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()

def norm01(x):
    x = x.astype(float); x = x - x.min()
    m = x.max(); return x / m if m > 0 else x

def process_image(file_path, control_baselines, outdir, nuclei_count=None,
                  selem_radius=50, positive_method="otsu", positive_percentile=80):
    fname = os.path.basename(file_path)
    session = get_max_number(fname)
    try:
        img = tiff.imread(file_path)
        ch1_raw, ch2_raw, dapi = img[0], img[1], img[2]
        ch1 = subtract_background(ch1_raw, selem_radius=selem_radius)
        ch2 = subtract_background(ch2_raw, selem_radius=selem_radius)
        tissue = make_tissue_mask(ch1, ch2, dapi)
        nuc, cyto = create_nucleus_and_cyto_masks(dapi, tissue)
        pos1_n = positive_mask(ch1, nuc, method=positive_method, percentile=positive_percentile)
        pos2_n = positive_mask(ch2, nuc, method=positive_method, percentile=positive_percentile)
        pos1_c = positive_mask(ch1, cyto, method=positive_method, percentile=positive_percentile)
        pos2_c = positive_mask(ch2, cyto, method=positive_method, percentile=positive_percentile)
        union_n = nuc & (pos1_n | pos2_n)
        union_c = cyto & (pos1_c | pos2_c)
        pearson_n = pearson_on_region(ch1, ch2, union_n)
        pearson_c = pearson_on_region(ch1, ch2, union_c)
        M1_n, M2_n = manders_coefficients(ch1, ch2, nuc, pos1_n, pos2_n)
        M1_c, M2_c = manders_coefficients(ch1, ch2, cyto, pos1_c, pos2_c)
        dice_n, jacc_n = dice_jaccard(pos1_n, pos2_n, nuc)
        dice_c, jacc_c = dice_jaccard(pos1_c, pos2_c, cyto)
        iwc_n, num_n = intensity_weighted_coloc(ch1, ch2, nuc, pos_mask=(pos1_n | pos2_n))
        iwc_c, num_c = intensity_weighted_coloc(ch1, ch2, cyto, pos_mask=(pos1_c | pos2_c))
        iwc_norm_ratio_n = normalized_iwc_ratio(ch1, ch2, nuc, control_baselines.get("nucleus"), pos_mask=(pos1_n | pos2_n))
        iwc_norm_ratio_c = normalized_iwc_ratio(ch1, ch2, cyto, control_baselines.get("cyto"),    pos_mask=(pos1_c | pos2_c))
        per_cell_overlap_n = (num_n / max(nuclei_count, 1)) if nuclei_count is not None else np.nan
        per_cell_overlap_c = (num_c / max(nuclei_count, 1)) if nuclei_count is not None else np.nan
        out_png = os.path.join(outdir, "qa_plots", session, os.path.splitext(fname)[0] + ".png")
        plot_filtered_visualization(out_png, norm01(ch1), norm01(ch2), dapi * nuc, pos1_n, pos2_n, pos_hi_union=(pos1_n & pos2_n))
        return {
            "MAX_NUMBER": session,
            "File": fname,
            "Path": file_path,
            "Region_N_Pixels": int(union_n.sum()),
            "Region_C_Pixels": int(union_c.sum()),
            "Pearson_Nucleus": pearson_n,
            "Pearson_Cyto": pearson_c,
            "Manders_M1_Nucleus": M1_n,
            "Manders_M2_Nucleus": M2_n,
            "Manders_M1_Cyto": M1_c,
            "Manders_M2_Cyto": M2_c,
            "Dice_Nucleus": dice_n,
            "Jaccard_Nucleus": jacc_n,
            "Dice_Cyto": dice_c,
            "Jaccard_Cyto": jacc_c,
            "IWC_Nucleus_%": iwc_n,
            "IWC_Cyto_%": iwc_c,
            "IWC_NormRatio_Nucleus": iwc_norm_ratio_n,
            "IWC_NormRatio_Cyto": iwc_norm_ratio_c,
            "Nuclei_Count": nuclei_count if nuclei_count is not None else np.nan,
            "OverlapIntensityPerCell_Nucleus": per_cell_overlap_n,
            "OverlapIntensityPerCell_Cyto": per_cell_overlap_c,
        }
    except Exception as e:
        logging.error(f"Error processing {fname}: {e}")
        return {}

def main():
    ap = argparse.ArgumentParser(description="Colocalization pipeline (reads Cellpose counts CSV; scans directories).")
    ap.add_argument("--root_dir", required=True, help="Top-level directory to scan (recursively)")
    ap.add_argument("--outdir", default="coloc_results", help="Output directory")
    ap.add_argument("--cellpose_csv", required=True, help="CSV from cellpose_count.py")
    ap.add_argument("--join_on", choices=["File", "MAX_NUMBER"], default="MAX_NUMBER")
    ap.add_argument("--selem_radius", type=int, default=50)
    ap.add_argument("--positive_method", choices=["otsu", "percentile"], default="otsu")
    ap.add_argument("--positive_percentile", type=int, default=80)
    ap.add_argument("--dapi_index", type=int, default=2, help="Channel index for DAPI")
    # optional filters
    ap.add_argument("--include", nargs="*", default=None, help="Only include paths containing any of these substrings")
    ap.add_argument("--exclude", nargs="*", default=None, help="Exclude paths containing any of these substrings")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # find images
    file_paths = list_image_files(args.root_dir, include=args.include, exclude=args.exclude)
    if not file_paths:
        print("No images found. Check --root_dir/filters.")
        return

    # control baselines from hpg-only controls discovered in the tree
    control_baselines = compute_control_baselines(file_paths, selem_radius=args.selem_radius)

    # nuclei counts
    cc = pd.read_csv(args.cellpose_csv)
    key_col = args.join_on
    if key_col not in cc.columns:
        if "File" in cc.columns and key_col == "MAX_NUMBER":
            cc["MAX_NUMBER"] = cc["File"].map(lambda f: get_max_number(os.path.basename(str(f))))
        else:
            raise ValueError(f"Cellpose CSV must contain '{key_col}' column.")
    nuclei_col = "nuclei_sam" if "nuclei_sam" in cc.columns else ("Nuclei" if "Nuclei" in cc.columns else None)
    if nuclei_col is None:
        raise ValueError("Could not find nuclei count column ('nuclei_sam' or 'Nuclei').")
    counts_map = {str(r[key_col]): int(r[nuclei_col]) for _, r in cc.iterrows()}

    rows = []
    for fp in file_paths:
        fname = os.path.basename(fp)
        session = get_max_number(fname)
        key = session if key_col == "MAX_NUMBER" else fname
        nuclei_count = counts_map.get(str(key), None)

        # read image once to validate shape & channel order; pass through processor
        try:
            img = tiff.imread(fp)
            if img.ndim < 3:
                logging.error(f"Skipping {fname}: expected (C, H, W) image.")
                continue
            # we re-read inside process_image for clarity; alternatively, pass ch1/ch2/dapi here
        except Exception as e:
            logging.error(f"Read failed {fname}: {e}")
            continue

        res = process_image(
            file_path=fp,
            control_baselines=control_baselines,
            outdir=args.outdir,
            nuclei_count=nuclei_count,
            selem_radius=args.selem_radius,
            positive_method=args.positive_method,
            positive_percentile=args.positive_percentile,
        )
        if res:
            rows.append(res)

    df = pd.DataFrame(rows)
    out_csv = os.path.join(args.outdir, "colocalization_results.csv")
    df.to_csv(out_csv, index=False)
    logging.info(f"Saved results to {out_csv}")
    print(f"Done. Wrote: {out_csv}")

if __name__ == "__main__":
    main()
