import argparse
import pathlib
import os
import sys
from tqdm import tqdm
import numpy as np
from PIL import Image
import torch
from torchvision.io import ImageReadMode, read_image
from skimage.filters import threshold_multiotsu
from skimage.exposure import histogram
from skimage.morphology import binary_erosion, disk
import kornia

def median_blur(sem, kernel_size: int) -> torch.Tensor:
    return kornia.filters.median_blur(sem.unsqueeze(0), kernel_size).squeeze(0)

def threshold(sem, threshold: int) -> torch.Tensor:
    thresh = torch.zeros_like(sem)
    thresh[sem > threshold] = 255
    return thresh

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=pathlib.Path, help="Input directory")
    parser.add_argument("output_dir", type=pathlib.Path, help="Output directory")
    parser.add_argument("-m", "--median_blur", type=int, default=1, help="Median blur kernel size")
    
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ipaths = sorted(list(args.input_dir.glob("*.png")))
    if not ipaths:
        print("No PNG files found")
        return

    # 1. Compute Histogram (Otsu)
    hist = np.zeros(256)
    for ipath in tqdm(ipaths, desc="Collecting histogram"):
        sem = np.array(Image.open(ipath), dtype=np.uint8)
        sem_hist, _ = histogram(sem, source_range='dtype')
        hist += sem_hist

    try:
        # Multi-otsu with 3 classes (Background, Tracks, Vias)
        # thresholds returned are [thresh1, thresh2]
        thresholds = threshold_multiotsu(hist=hist, classes=3)
        track_thresh = thresholds[0]
        via_thresh = thresholds[1]
        print(f"Otsu thresholds: {track_thresh} (tracks), {via_thresh} (vias)")
    except Exception as e:
        print(f"Otsu failed: {e}. Fallback to defaults.")
        track_thresh = 67
        via_thresh = 157

        # 2. Generate Masks
    for ipath in tqdm(ipaths, desc="Generating masks"):
        # Load image
        sem = read_image(str(ipath), mode=ImageReadMode.GRAY).to(device, torch.float)
        
        # Blur
        if args.median_blur > 1:
            sem = median_blur(sem, args.median_blur)
            
        # Create separate masks
        # Tracks: > track_thresh but ideally excluding vias if we want pure separation, 
        # but typically tracks+vias is valid connectivity. 
        # However, for visual replication of "dots on lines":
        # Tracks mask = (sem > track_thresh)
        # Vias mask = (sem > via_thresh)
        
        mtracks = threshold(sem, track_thresh)
        mvias = threshold(sem, via_thresh)
        
        # Save Tracks (Green)
        # We subtract Vias from Tracks? Or keep them? 
        # The reference `gen_pseudo_masks.py` did `mtracks = mtracks - mvias`.
        # Let's try that for cleaner separation.
        mtracks_exclusive = (mtracks - mvias).clamp(0, 255)
        
        # ERODE to thin the lines (User request: "fat lines")
        # Convert to numpy boolean for skimage
        # Ensure it's 2D (Squeeze channel dim if present, typically [1, H, W] -> [H, W])
        tracks_bool = mtracks_exclusive.squeeze().cpu().numpy() > 128
        # Use disk(1) or disk(2) depending on how fat they are. 
        # User complained disk(1) was still fat. Increasing to disk(4).
        # Fix: Potrace traces Black. We want to trace Tracks. So Tracks must be Black (0).
        # Previously we saved Tracks as White (255), causing Potrace to trace Background (Black).
        # Erosion of White Tracks = Thinner White -> Fatter Black Background -> Fatter Lines.
        # Correct logic: Erode White Tracks -> Save as Black.
        # User feedback: "Dots merged" (Blur issue?) and "Too thin" (Erosion issue).
        # Fix 1: Reduced blur to 1 (None).
        # User feedback: "Dots merged" (Blur issue?) and "Too thin" (Erosion issue).
        # Fix 1: Reduced blur to 1 (None).
        # User feedback: "Red dots double/small" (Via fragmentation) & "Thick parts" (Track noise/var).
        # Fix Vias: Dilation to merge fragments and enlarge.
        # Fix Tracks: Opening (Erode->Dilate) to clean, then Dilate to thicken.
        
        from skimage.morphology import binary_dilation, binary_opening, binary_closing
        
        # User feedback: "Single red dot" (Over-merged Vias) & "Still fat" (Tracks 18.5px).
        # Fix Vias: Reduce dilation. disk(5) -> disk(2).
        # Fix Tracks: 18.5px is fat. 13px is thin. Need ~15-16px.
        # Integer dilation is too coarse (0->13px, 1->17.8px).
        # Solution: Fractional Dilation using Distance Transform.
        # Target: Add ~1.2px radius => +2.4px width. 13 + 2.4 = 15.4px.
        
        # User feedback: "Still fat" (17.2px). "Red dots are 2 but merged".
        # Fix Vias: Remove dilation. Let them be separated as per raw segmentation.
        # Fix Tracks: 17.2px is "fat". 13px was "thin".
        # Target: ~14.5px.
        # Strategy: EDT <= 0.5. Adds 0.5px radius (1px width) to base ~13px.
        
        # User feedback: "Still fat" (16.7px). "Merged dots" (Closing 3).
        # Fix Vias: Revert to RAW. Closing merged them. Separation is key.
        # Fix Tracks: 16.7px is fat. 13px is thin. Target ~15px.
        # Strategy: Increase Analog Threshold. 0.75 -> 0.9.
        # This erodes the edges of the dilated mask further.
        
        from skimage.morphology import binary_dilation, binary_closing, disk
        from scipy.ndimage import zoom
        
        # 1. Process Vias (Raw)
        vias_bool = mvias.squeeze().cpu().numpy() > 128
        # No morphology. Prioritize separation.
        mask_vias_np = (vias_bool * 255).astype(np.uint8)
        Image.fromarray(mask_vias_np).save(args.output_dir / f"{ipath.stem}_vias.png")

        # 2. Process Tracks (Analog Downscale)
        # Upscale 4x
        tracks_4x = zoom(tracks_bool, 4.0, order=0)
        
        # Dilate(disk 2) in 4x space.
        tracks_dilated_4x = binary_dilation(tracks_4x, disk(2))
        
        # Downscale 0.25x with LINEAR interpolation (order=1)
        tracks_float = zoom(tracks_dilated_4x.astype(float), 0.25, order=1)
        
        # Threshold > 0.9.
        # Very aggressive thinning of the dilated mask.
        tracks_subpix = tracks_float > 0.9
        
        # Invert: Tracks (True) -> 0 (Black). Background (False) -> 255 (White).
        mask_tracks_np = ((~tracks_subpix) * 255).astype(np.uint8)
        
        # Ensure binary
        tracks_subpix = tracks_subpix > 0.5
        
        # Invert: Tracks (True) -> 0 (Black). Background (False) -> 255 (White).
        mask_tracks_np = ((~tracks_subpix) * 255).astype(np.uint8)
        if mask_tracks_np.ndim == 3: mask_tracks_np = mask_tracks_np.squeeze()
        Image.fromarray(mask_tracks_np).save(args.output_dir / f"{ipath.stem}_tracks.png")

        # Save Vias (Red)
        mask_vias_np = mvias.cpu().numpy().astype(np.uint8)
        if mask_vias_np.ndim == 3: mask_vias_np = mask_vias_np.squeeze()
        Image.fromarray(mask_vias_np).save(args.output_dir / f"{ipath.stem}_vias.png")
        
    print("Segmentation complete (Tracks & Vias separated).")

if __name__ == "__main__":
    main()
