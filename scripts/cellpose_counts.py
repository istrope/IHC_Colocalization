#!/usr/bin/env python3
# cellpose_count.py
import os
import re
import argparse
import logging
import numpy as np
import pandas as pd
import tifffile as tiff
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from cellpose import models, plot

logging.basicConfig(
    filename="cellpose_count.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

def get_max_number(text: str) -> str:
    """Extract 'MAX_###' token from a filename or path."""
    m = re.search(r"(MAX_\d+)", os.path.basename(text))
    return m.group(1) if m else "UNKNOWN"

def list_image_files(root_dir, include=None, exclude=None, exts=(".tif", ".tiff")):
    """Recursively list image files, with optional substring include/exclude filters."""
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

def normalize_to_uint8(img: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0,255] uint8."""
    img = img.astype(np.float32)
    img -= img.min()
    m = img.max()
    if m > 0:
        img /= m
    return (img * 255).astype(np.uint8)

def count_nuclei_with_cellpose(dapi_img: np.ndarray, model: models.CellposeModel, params: dict):
    """Run Cellpose on a DAPI channel with tuned parameters; return (count, mask, flows)."""
    work = normalize_to_uint8(dapi_img)
    sigma = float(params.get("sigma_dapi", 0))
    if sigma > 0:
        work = gaussian_filter(work, sigma=sigma)

    eval_kwargs = dict(
        diameter=float(params.get("diameter", 75)),
        flow_threshold=float(params.get("flow_threshold", 0.13039557764547482)),
        cellprob_threshold=float(params.get("cellprob_threshold", -0.14824550053306051)),
        min_size=int(params.get("min_size", 16)),
        augment=bool(params.get("augment", True)),
        tile_overlap=float(params.get("tile_overlap", 0.3225473697868176)),
        normalize=True
    )
    masks, flows, styles = model.eval([work], **eval_kwargs)
    lbl = masks[0] if isinstance(masks, (list, tuple)) else masks
    return int(lbl.max()), lbl, flows

def save_qa_png(out_png: str, dapi: np.ndarray, mask: np.ndarray, flows):
    """Save a QA segmentation visualization."""
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig = plt.figure(figsize=(12, 5))
    flow = flows[0] if isinstance(flows, (list, tuple)) else flows
    flow_to_plot = flow[0] if isinstance(flow, (list, tuple)) else flow
    plot.show_segmentation(fig, dapi, mask, flow_to_plot)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def main():
    ap = argparse.ArgumentParser(description="Run Cellpose (nuclei model) on DAPI to count nuclei; save CSV and QA PNGs.")
    ap.add_argument("--root_dir", required=True, help="Top-level directory to scan (recursively)")
    ap.add_argument("--out_csv", default="cell_counts.csv", help="Output CSV path")
    ap.add_argument("--qa_dir", default="cellpose_masks", help="Directory to save QA PNGs")
    ap.add_argument("--gpu", action="store_true", help="Use GPU for Cellpose")
    ap.add_argument("--dapi_index", type=int, default=2, help="Channel index for DAPI in (C,H,W) arrays")
    # optional substring filters
    ap.add_argument("--include", nargs="*", default=None, help="Only include paths containing any of these substrings")
    ap.add_argument("--exclude", nargs="*", default=None, help="Exclude paths containing any of these substrings")
    # tuned params
    ap.add_argument("--sigma_dapi", type=float, default=4.871771992396551)
    ap.add_argument("--diameter", type=float, default=75)
    ap.add_argument("--flow_threshold", type=float, default=0.13039557764547482)
    ap.add_argument("--cellprob_threshold", type=float, default=-0.14824550053306051)
    ap.add_argument("--min_size", type=int, default=16)
    ap.add_argument("--augment", action="store_true", default=True)
    ap.add_argument("--tile_overlap", type=float, default=0.3225473697868176)
    args = ap.parse_args()

    # collect params
    params = {
        "sigma_dapi": args.sigma_dapi,
        "diameter": args.diameter,
        "flow_threshold": args.flow_threshold,
        "cellprob_threshold": args.cellprob_threshold,
        "min_size": args.min_size,
        "augment": args.augment,
        "tile_overlap": args.tile_overlap,
    }

    # built-in nuclei model (no weights.pt needed)
    model = models.CellposeModel(pretrained_model="nuclei", gpu=args.gpu)

    files = list_image_files(args.root_dir, include=args.include, exclude=args.exclude)
    if not files:
        print("No images found. Check --root_dir and filters.")
        return

    rows = []
    for fp in files:
        try:
            img = tiff.imread(fp)
            if img.ndim < 3:
                raise ValueError(f"Expected image with shape (C,H,W); got {img.shape}.")
            dapi = img[args.dapi_index]
            count, mask, flows = count_nuclei_with_cellpose(dapi, model, params)

            fname = os.path.basename(fp)
            session = get_max_number(fname)
            rows.append({"File": fname, "Path": fp, "MAX_NUMBER": session, "nuclei_sam": count})

            out_png = os.path.join(args.qa_dir, session, os.path.splitext(fname)[0] + ".png")
            save_qa_png(out_png, dapi, mask, flows)
            print(f"[OK] {fname} -> nuclei: {count}")

        except Exception as e:
            logging.error(f"Cellpose failed on {fp}: {e}")
            print(f"[ERR] {fp} :: {e}")

    df = pd.DataFrame(rows).sort_values(by=["MAX_NUMBER", "File"])
    df.to_csv(args.out_csv, index=False)
    print(f"Saved counts: {args.out_csv}")

if __name__ == "__main__":
    main()
