"""
CanSat Ground Station - Image Processing Module
Author: Aerospace Software Engineer
Date: July 2026

Calculates vegetation indices (VARI, ExG) using vectorized NumPy matrix operations,
performs canopy coverage classification, generates colormapped heatmaps, and overlays
a professional Head-Up Display (HUD) with real-time flight metrics and telemetry.
"""

import time
import numpy as np
import cv2

def calculate_vari(R, G, B):
    """
    Computes the Visible Atmospherically Resistant Index (VARI).
    Formula: (G - R) / (G + R - B + 1e-6)
    Ranges from -1.0 to +1.0. High greenness yields values closer to 1.0.
    """
    denominator = G + R - B + 1e-6
    return (G - R) / denominator

def calculate_exg(R, G, B):
    """
    Computes the Excess Green Index (ExG).
    Formula: 2G - R - B
    Highlights green vegetation from soil and residue using raw pixel differences.
    """
    return 2.0 * G - R - B

def draw_hud(img, coverage_pct, fps, label):
    """
    Draws a semi-transparent HUD overlay on the image containing telemetry metadata
    and visual warning banners if vegetation cover drops below safe ecological thresholds.
    """
    h, w, _ = img.shape
    overlay = img.copy()

    # 1. Top banner: Dark translucent strip for stream label and UTC timestamp
    cv2.rectangle(overlay, (0, 0), (w, 40), (20, 20, 20), -1)

    # 2. Bottom banner: Color-coded telemetry and status alert
    # Canopy coverage threshold: Alert if less than 20% (indicates soil erosion risk or bare terrain)
    is_alert = coverage_pct < 20.0
    if is_alert:
        # BGR: Crimson Red warning banner for low coverage
        banner_color = (15, 15, 180)
        status_text = "ALERT: LOW CANOPY COVER (EROSION RISK)"
    else:
        # BGR: Forest Green nominal banner
        banner_color = (25, 120, 25)
        status_text = "STATUS: NOMINAL CANOPY DENSITY"

    cv2.rectangle(overlay, (0, h - 45), (w, h), banner_color, -1)

    # Blend the overlay at 75% opacity to allow camera pixels to show underneath the banners
    cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)

    # 3. Write HUD Text Elements
    # Current timestamp in UTC / Local format
    time_str = time.strftime("%Y-%m-%d %H:%M:%S")
    
    # Top text: Module Title
    cv2.putText(img, f"CANSAT TELEMETRY | {label}", (15, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, time_str, (w - 210, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    # Bottom text: Core sensor values and alert strings
    metrics_str = f"COVERAGE: {coverage_pct:.2f}% | FPS: {fps:.1f} | {status_text}"
    cv2.putText(img, metrics_str, (15, h - 17),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    return img

def process_frame(frame, actual_fps, threshold_vari=0.1, threshold_exg=15.0):
    """
    Core image processing pipeline. Resolves vegetation indices, extracts masks,
    applies colormaps, and runs HUD overlays.
    
    Args:
        frame (np.ndarray): Input image in OpenCV BGR format (800x600 px).
        actual_fps (float): Calculated ingest frame rate.
        threshold_vari (float): Cut-off for green pixels using VARI. Default 0.1.
        threshold_exg (float): Cut-off for green pixels using ExG. Default 15.0.

    Returns:
        processed_rgb (np.ndarray): Original image with slight classification green-overlay and HUD.
        processed_heatmap (np.ndarray): Heatmap of vegetation density (VARI) and HUD.
        metrics (dict): Dict of statistics (coverage %, mean VARI, mean ExG).
    """
    # Extract channels and convert to float32 for calculations (prevents overflow/underflow)
    B = frame[:, :, 0].astype(np.float32)
    G = frame[:, :, 1].astype(np.float32)
    R = frame[:, :, 2].astype(np.float32)

    # 1. Compute Indices
    vari = calculate_vari(R, G, B)
    exg = calculate_exg(R, G, B)

    # 2. Segment Vegetation
    # Pixel is classified as healthy vegetation if both indices exceed their thresholds
    veg_mask = (vari > threshold_vari) & (exg > threshold_exg)
    
    total_pixels = frame.shape[0] * frame.shape[1]
    veg_pixels = np.sum(veg_mask)
    coverage_pct = (veg_pixels / total_pixels) * 100.0

    # Calculate regional indices statistics
    mean_vari = float(np.mean(vari))
    mean_exg = float(np.mean(exg))

    # 3. Generate Colormapped Heatmap
    # Normalize VARI values from typical range [-0.2, 0.8] to [0, 255]
    vari_normalized = np.clip((vari - (-0.2)) / (0.8 - (-0.2)) * 255.0, 0, 255).astype(np.uint8)
    # Apply COLORMAP_JET (blue=low greenness/soil, red=dense healthy canopy)
    heatmap = cv2.applyColorMap(vari_normalized, cv2.COLORMAP_JET)

    # 4. Highlight segmented vegetation on the RGB frame (translucent green mask)
    highlight_rgb = frame.copy()
    highlight_rgb[veg_mask] = highlight_rgb[veg_mask] * 0.7 + np.array([0, 255, 0]) * 0.3
    highlight_rgb = np.clip(highlight_rgb, 0, 255).astype(np.uint8)

    # 5. Apply HUD telemetry layers
    processed_rgb = draw_hud(highlight_rgb, coverage_pct, actual_fps, "RGB FIELD VIEW")
    processed_heatmap = draw_hud(heatmap, coverage_pct, actual_fps, "VEGETATION INDEX (VARI)")

    return processed_rgb, processed_heatmap, {
        "coverage_pct": round(coverage_pct, 2),
        "mean_vari": round(mean_vari, 4),
        "mean_exg": round(mean_exg, 2),
        "is_alert": bool(is_alert := coverage_pct < 20.0)
    }
