import os
import glob
import argparse
from svg_vector_stitcher import SVGVectorStitcher

def main():
    parser = argparse.ArgumentParser(description="Stitch SVGs from m2 directory")
    parser.add_argument("--limit", type=int, default=20, help="Number of files to stitch (default: 20)")
    parser.add_argument("--method", type=str, default="loftr", help="Matching method (loftr, lightglue, disk)")
    args = parser.parse_args()

    # Path to m2 directory
    m2_dir = "/app/data/ai-stitching-model/m2"
    output_path = "/app/data/ai-stitching-model/panorama_m2.svg"
    
    # Get all SVG files
    svg_files = sorted(glob.glob(os.path.join(m2_dir, "*.svg")))
    
    if not svg_files:
        print(f"No SVG files found in {m2_dir}")
        return

    print(f"Found {len(svg_files)} SVG files.")
    
    # Initialize stitcher
    print(f"Initializing stitcher with method: {args.method}")
    stitcher = SVGVectorStitcher(
        use_transformer=True, # Fallback
        use_raster_matching=True,
        raster_method=args.method,
        layout_mode='grid'
    )
    
    # Run stitching
    subset_files = svg_files[:args.limit]
    print(f"Stitching {len(subset_files)} files...")
    
    success = stitcher.create_panorama_svg(subset_files, output_path)
    
    if success:
        print(f"Successfully created panorama at {output_path}")
    else:
        print("Failed to create panorama")

if __name__ == "__main__":
    main()
