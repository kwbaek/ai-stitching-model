
import argparse
import cairosvg
import numpy as np
from PIL import Image
import io
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt
import sys

def rasterize_svg(svg_path, scale=1.0):
    """Rasterize SVG to numpy array (H, W, 3)"""
    png_data = cairosvg.svg2png(url=str(svg_path), scale=scale)
    image = Image.open(io.BytesIO(png_data)).convert("RGB")
    return np.array(image)

def get_track_width_stats(image):
    """
    Compute width stats for 'Lime' tracks.
    Assumes tracks are green (G > R and G > B).
    """
    # Simple color threshold for Lime (0, 255, 0) logic
    # Just take Green channel dominance
    r, g, b = image[:,:,0], image[:,:,1], image[:,:,2]
    # Lime is approx R=0, G=255, B=0. Allow some antialiasing slack.
    # Mask where Green is significant and larger than Red/Blue
    mask = (g > 100) & (g > r + 30) & (g > b + 30)
    
    if not np.any(mask):
        return 0.0, 0.0

    # Distance Transform
    # dt values are distance to nearest background (0). 
    # For a line of width W, the center pixels have value W/2.
    dt = distance_transform_edt(mask)
    
    # Skeletonize to find centerlines
    skeleton = skeletonize(mask)
    
    if not np.any(skeleton):
        return 0.0, 0.0
        
    # Sample DT at skeleton locations
    widths_half = dt[skeleton]
    widths = widths_half * 2.0
    
    return np.mean(widths), np.std(widths)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ref_svg", help="Reference SVG file")
    parser.add_argument("gen_svg", help="Generated SVG file")
    args = parser.parse_args()

    # 1. Rasterize
    # Use scale > 1 for better sub-pixel estimation if needed, but 1.0 matches pixel grid
    scale = 1.0 
    print(f"Rasterizing {args.ref_svg}...")
    ref_img = rasterize_svg(args.ref_svg, scale=scale)
    print(f"Rasterizing {args.gen_svg}...")
    gen_img = rasterize_svg(args.gen_svg, scale=scale)

    # 2. Compute Stats
    ref_mean, ref_std = get_track_width_stats(ref_img)
    gen_mean, gen_std = get_track_width_stats(gen_img)

    print("-" * 30)
    print(f"Reference Width: {ref_mean:.2f} px (std: {ref_std:.2f})")
    print(f"Generated Width: {gen_mean:.2f} px (std: {gen_std:.2f})")
    
    diff = gen_mean - ref_mean
    print(f"Difference (Gen - Ref): {diff:.2f} px")
    print("-" * 30)
    
    # Recommendation
    # Erosion removes pixels from boundaries. 
    # Erosion radius R reduces width by 2*R.
    # We want Gen to match Ref. 
    # If Diff > 0 (Gen is thicker), we need MORE erosion.
    # Extra erosion needed (approx) = Diff / 2.
    
    if abs(diff) > 0.5:
        correction = diff / 2.0
        print(f"RECOMMENDATION: Adjust Erosion Radius by +{correction:.2f}")
    else:
        print("RECOMMENDATION: Width matches within tolerance.")

if __name__ == "__main__":
    main()
