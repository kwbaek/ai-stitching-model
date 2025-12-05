import os
import glob
import argparse
from pathlib import Path
import json
from svg_vector_stitcher import SVGVectorStitcher

def main():
    parser = argparse.ArgumentParser(description="Stitch SVGs from m2 directory")
    parser.add_argument("--limit", type=int, default=20, help="Number of files to stitch (default: 20)")
    parser.add_argument("--method", type=str, default="loftr", help="Matching method (loftr, lightglue, disk)")
    parser.add_argument('--show-labels', action='store_true', help='Show filename labels in SVG')
    parser.add_argument('--show-borders', action='store_true', help='Show borders around tiles in SVG')
    parser.add_argument('--vectorize', action='store_true', help='Run vectorization from PNGs first')
    
    args = parser.parse_args()
    
    # Pipeline Configuration
    base_dir = Path('/app/data/ai-stitching-model')
    png_dir = base_dir / 'dataset/sems/m2'
    vector_dir = base_dir / 'dataset/vectorized_m2'
    
    # 1. Vectorization Stage
    if args.vectorize:
        print("\n=== Stage 1: Vectorization ===")
        from vectorize_images import vectorize_images
        
        if not png_dir.exists():
            print(f"Error: PNG directory not found: {png_dir}")
            return
            
        vectorize_images(str(png_dir), str(vector_dir))
        
        # Update input directory for stitching to use the newly vectorized files
        # Note: The Stitcher expects a directory relative to base or absolute.
        # We'll use the relative path 'dataset/vectorized_m2' if the stitcher supports it,
        # or we might need to adjust how the stitcher finds files.
        # Looking at SVGVectorStitcher, it takes 'data_dir' and looks for 'm2/*.svg' inside it if we pass 'm2'.
        # Let's check how we pass the directory.
        
    # 2. Stitching Stage
    print("\n=== Stage 2: Stitching ===")
    
    # Determine input directory for stitching
    if args.vectorize:
        # If we vectorized, we want to stitch the files in dataset/vectorized_m2
        # The current stitcher implementation might be hardcoded to look in 'm2' subdirectory of data_dir.
        # We need to be careful here.
        # Let's assume we pass the full path or relative path that works.
        # If we pass 'dataset/vectorized_m2' as the dataset name?
        pass 

    # Update progress for stitching start
    with open('progress_pipeline.json', 'w') as f:
        json.dump({"stage": "Stitching", "status": "Starting stitching process..."}, f)

    stitcher = SVGVectorStitcher(
        raster_method=args.method,
        layout_mode='grid',
        show_labels=args.show_labels,
        show_borders=args.show_borders
    )
    
    # Determine input files
    input_dir = vector_dir if args.vectorize else base_dir / 'dataset/labels/m2'
    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        return
        
    svg_files = sorted(glob.glob(str(input_dir / "*.svg")))
    if not svg_files:
        print(f"Error: No SVG files found in {input_dir}")
        return
        
    print(f"Found {len(svg_files)} SVG files in {input_dir}")
    
    # Run stitching
    subset_files = svg_files[:args.limit]
    print(f"Stitching {len(subset_files)} files...")
    
    output_path = str(base_dir / 'panorama_m2.svg')
    success = stitcher.create_panorama_svg(subset_files, output_path)
    
    if success:
        abs_path = os.path.abspath(output_path)
        if os.path.exists(abs_path):
            file_size = os.path.getsize(abs_path)
            print(f"SUCCESS: Successfully created panorama at {abs_path}")
            print(f"File size: {file_size} bytes")
        else:
            print(f"ERROR: Stitcher reported success but file not found at {abs_path}")
            return
    else:
        print("Failed to create panorama")
        return

    # 3. GDS Export Stage
    print("\n=== Stage 3: GDS Export ===")
    with open('progress_pipeline.json', 'w') as f:
        json.dump({"stage": "GDS Export", "status": "Converting to GDSII..."}, f)
        
    from svg_to_gds import svg_to_gds
    output_gds = output_path.replace('.svg', '.gds')
    svg_to_gds(output_path, output_gds)
    
    with open('progress_pipeline.json', 'w') as f:
        json.dump({"stage": "Complete", "status": "Pipeline finished successfully!"}, f)

if __name__ == "__main__":
    main()
