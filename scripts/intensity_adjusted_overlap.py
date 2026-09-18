import re
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
import pandas as pd
import os
import logging
from skimage import filters, morphology
from skimage.morphology import disk, opening
from scipy.stats import pearsonr
from collections import defaultdict

# Configure logging
logging.basicConfig(
    filename="colocalization_analysis.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Function to extract MAX_NUMBER (imaging session)
def get_max_number(filename):
    match = re.match(r"(MAX_\d+)", filename)
    return match.group(1) if match else "UNKNOWN"

# Function to parse file structure
def parse_file_structure(file_path):
    session_files = defaultdict(lambda: defaultdict(list))
    current_directory = ""
    with open(file_path, "r") as f:
        file_contents = f.readlines()
    for line in file_contents:
        line = line.strip().strip("'")
        if line.endswith("/"):
            current_directory = line
        else:
            full_file_path = f'{current_directory}{line}'
            session = get_max_number(line)
            if session != 'UNKNOWN':
                session_files[session]['files'].append(full_file_path)
    return session_files

# Subtract local background using morphological opening.
def subtract_background(image, selem_radius=50):
    selem = disk(selem_radius)
    background = opening(image, selem)
    corrected = image - background
    corrected[corrected < 0] = 0
    return corrected

# Apply thresholding per sample using Otsu's method
def apply_threshold(image, threshold):
    return image * (image > threshold)

# Normalize per image after filtering
def normalize_per_image(image):
    max_intensity = np.max(image)
    return image / (max_intensity + 1e-6)

# Create nucleus and cytoplasmic masks based on DAPI channel
def create_nucleus_and_cyto_masks(dapi_channel):
    dapi_threshold = filters.threshold_otsu(dapi_channel)
    nucleus_mask = dapi_channel > dapi_threshold
    cyto_mask = ~nucleus_mask
    cyto_mask = morphology.remove_small_holes(cyto_mask, area_threshold=50)
    return nucleus_mask, cyto_mask

# Compute Pearson Correlation Coefficient between two channels
def compute_pearson_correlation(ch1, ch2):
    return pearsonr(ch1.flatten(), ch2.flatten())[0]

# Compute intensity weighted colocalization.
# If reference_total_intensity is provided (from hpg_only controls), use that instead
def compute_intensity_weighted_colocalization(ch1, ch2, compartment_mask, min_intensity_fraction=0.2, reference_total_intensity=None):
    comp_ch1 = ch1 * compartment_mask
    comp_ch2 = ch2 * compartment_mask

    if reference_total_intensity is None:
        total_intensity = np.sum(comp_ch1) + np.sum(comp_ch2)
    else:
        total_intensity = reference_total_intensity

    if total_intensity == 0:
        return np.nan

    # Apply a secondary intensity cutoff within the compartment
    cutoff_ch1 = min_intensity_fraction * np.max(ch1 * compartment_mask)
    cutoff_ch2 = min_intensity_fraction * np.max(ch2 * compartment_mask)
    high_signal_mask_ch1 = ch1 >= cutoff_ch1
    high_signal_mask_ch2 = ch2 >= cutoff_ch2
    overlap_mask = np.logical_and(high_signal_mask_ch1, high_signal_mask_ch2)
    overlap_mask = np.logical_and(overlap_mask, compartment_mask)
    overlap_intensity = np.sum(ch1 * overlap_mask) + np.sum(ch2 * overlap_mask)
    weighted_percent = (overlap_intensity / total_intensity) * 100
    return weighted_percent

# Updated plotting function to include a secondary high-intensity overlay
def plot_filtered_visualization(file_path, ch1_norm, ch2_norm, dapi_filtered, ch1_thresh, ch2_thresh, min_intensity_fraction=0.2):
    plot_file = re.sub('Batch_1', 'intensity_adjusted_out', file_path)
    plot_file = re.sub('.tif', '.png', plot_file)
    os.makedirs(os.path.dirname(plot_file), exist_ok=True)
    
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    axes[0].imshow(ch1_norm, cmap='Reds')
    axes[0].set_title("Normalized Channel 1")
    
    axes[1].imshow(ch2_norm, cmap='Greens')
    axes[1].set_title("Normalized Channel 2")
    
    axes[2].imshow(dapi_filtered, cmap='Blues')
    axes[2].set_title("Filtered DAPI (Nucleus Mask)")
    
    overlay = np.dstack((ch1_thresh, ch2_thresh, np.zeros_like(ch1_thresh)))
    axes[3].imshow(overlay)
    axes[3].set_title("Colocalization Overlay")
    
    cutoff_ch1 = min_intensity_fraction * np.max(ch1_thresh)
    cutoff_ch2 = min_intensity_fraction * np.max(ch2_thresh)
    high_signal_mask = np.logical_and(ch1_thresh >= cutoff_ch1, ch2_thresh >= cutoff_ch2)
    secondary_overlay = np.dstack((ch1_thresh, ch2_thresh, np.zeros_like(ch1_thresh)))
    secondary_overlay[~high_signal_mask] = 0
    axes[4].imshow(secondary_overlay)
    axes[4].set_title("High-Intensity Coloc Overlay")
    
    plt.tight_layout()
    plt.savefig(plot_file)
    plt.close()
    logging.info(f"Saved filtered visualization: {plot_file}")

# Compute average control (hpg_only) total intensities for nucleus and cytosol
def compute_control_reference_intensity(session_files):
    control_intensities_nucleus = []
    control_intensities_cyto = []
    
    for session, data in session_files.items():
        for file in data["files"]:
            if "hpg only" in file.lower():
                try:
                    img = tiff.imread(file)
                    ch1_raw, ch2_raw, dapi = img[0], img[1], img[2]
                    ch1_corrected = subtract_background(ch1_raw, selem_radius=50)
                    ch2_corrected = subtract_background(ch2_raw, selem_radius=50)
                    ch1_threshold = filters.threshold_otsu(ch1_corrected)
                    ch2_threshold = filters.threshold_otsu(ch2_corrected)
                    ch1_thresh = apply_threshold(ch1_corrected, ch1_threshold)
                    ch2_thresh = apply_threshold(ch2_corrected, ch2_threshold)
                    nucleus_mask, cyto_mask = create_nucleus_and_cyto_masks(dapi)
                    
                    total_nucleus = np.sum(ch1_thresh * nucleus_mask) + np.sum(ch2_thresh * nucleus_mask)
                    total_cyto = np.sum(ch1_thresh * cyto_mask) + np.sum(ch2_thresh * cyto_mask)
                    control_intensities_nucleus.append(total_nucleus)
                    control_intensities_cyto.append(total_cyto)
                except Exception as e:
                    logging.error(f"Error processing control image {file}: {str(e)}")
    
    ref_nucleus = np.mean(control_intensities_nucleus) if control_intensities_nucleus else None
    ref_cyto = np.mean(control_intensities_cyto) if control_intensities_cyto else None
    return {'nucleus': ref_nucleus, 'cyto': ref_cyto}

# Process an individual image using the control reference intensities
def process_image(file_path, control_refs):
    filename = os.path.basename(file_path)
    session = get_max_number(filename)
    try:
        logging.info(f"Processing image: {filename}, Session: {session}")
        img = tiff.imread(file_path)
        ch1_raw, ch2_raw, dapi = img[0], img[1], img[2]
        
        # Background subtraction
        ch1_corrected = subtract_background(ch1_raw, selem_radius=50)
        ch2_corrected = subtract_background(ch2_raw, selem_radius=50)
        
        # Per-sample Otsu thresholds
        ch1_threshold = filters.threshold_otsu(ch1_corrected)
        ch2_threshold = filters.threshold_otsu(ch2_corrected)
        ch1_thresh = apply_threshold(ch1_corrected, ch1_threshold)
        ch2_thresh = apply_threshold(ch2_corrected, ch2_threshold)
        
        ch1_norm = normalize_per_image(ch1_thresh)
        ch2_norm = normalize_per_image(ch2_thresh)
        
        nucleus_mask, cyto_mask = create_nucleus_and_cyto_masks(dapi)
        pearson_corr = compute_pearson_correlation(ch1_thresh, ch2_thresh)
        
        # Use control references for normalization of the weighted colocalization
        percent_coloc_nucleus = compute_intensity_weighted_colocalization(
            ch1_thresh, ch2_thresh, nucleus_mask, min_intensity_fraction=0.2, 
            reference_total_intensity=control_refs['nucleus']
        )
        percent_coloc_cyto = compute_intensity_weighted_colocalization(
            ch1_thresh, ch2_thresh, cyto_mask, min_intensity_fraction=0.2, 
            reference_total_intensity=control_refs['cyto']
        )
        
        plot_filtered_visualization(file_path, ch1_norm, ch2_norm, dapi * nucleus_mask, ch1_thresh, ch2_thresh, min_intensity_fraction=0.2)
        
        return {
            "MAX_NUMBER": session,
            "File Basename": filename,
            "Pearson Correlation": pearson_corr,
            "Intensity Weighted Percent Coloc Nucleus": percent_coloc_nucleus,
            "Intensity Weighted Percent Coloc Cytoplasm": percent_coloc_cyto,
            "Ch1 Threshold": ch1_threshold,
            "Ch2 Threshold": ch2_threshold
        }
    except Exception as e:
        logging.error(f"Error processing image {filename}: {str(e)}")
        return {}

# Main workflow
file_path = "Image_files_controls.txt"
session_files = parse_file_structure(file_path)
# Compute control reference intensities from hpg_only files
control_refs = compute_control_reference_intensity(session_files)
results = []
for session, data in session_files.items():
    for file_path in data["files"]:
        result = process_image(file_path, control_refs)
        results.append(result)
df = pd.DataFrame(results)
df.to_csv("colocalization_results.csv", index=False)
logging.info("Saved results to colocalization_results.csv")
print("Colocalization analysis completed.")
