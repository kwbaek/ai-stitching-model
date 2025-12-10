import os
import glob
import argparse
from pathlib import Path
import json
import subprocess
from svg_vector_stitcher import SVGVectorStitcher

def update_status(stage, status, percent=0, current=0, total=0):
    """Helper to update pipeline status file"""
    data = {
        "stage": stage,
        "status": status,
        "percent": percent,
        "current": current,
        "total": total
    }
    with open('progress_pipeline.json', 'w') as f:
        json.dump(data, f)

def main():
    parser = argparse.ArgumentParser(description="Stitch SVGs from m2 directory")
    parser.add_argument("--limit", type=int, default=20, help="Number of files to stitch (default: 20)")
    parser.add_argument("--method", type=str, default="loftr", help="Matching method (loftr, lightglue, disk)")
    parser.add_argument('--show-labels', action='store_true', help='Show filename labels in SVG')
    parser.add_argument('--show-borders', action='store_true', help='Show borders around tiles in SVG')
    parser.add_argument('--vectorize', action='store_true', help='Run full pipeline (Segmentation -> Vectorization)')
    parser.add_argument('--use-manual-dir', action='store_true', help='Use manual upload directory')
    
    args = parser.parse_args()
    
    # Pipeline Configuration
    base_dir = Path('/app/data/ai-stitching-model')
    
    if args.use_manual_dir:
        print("Using Manual upload directory...")
        sem_dir = base_dir / 'dataset/sems/manual'
        mask_dir = base_dir / 'dataset/masks/manual'
        vector_dir = base_dir / 'dataset/vectorized_manual'
        output_name = 'manual_panorama'
        
        # Ensure directories exist
        mask_dir.mkdir(parents=True, exist_ok=True)
        vector_dir.mkdir(parents=True, exist_ok=True)
    else:
        # Default M2 dataset structure
        # Assuming raw SEMs might be in dataset/sems/m2 if we were to run full pipeline?
        # But existing code used 'png_dir' as input to vectorize.
        sem_dir = base_dir / 'dataset/sems/m2' 
        mask_dir = base_dir / 'dataset/masks/m2'
        vector_dir = base_dir / 'dataset/vectorized_m2'
        output_name = 'panorama_m2'

    # 1. Segmentation Stage
    if args.vectorize:
        print("\n=== Stage 1: Segmentation (Otsu) ===")
        update_status("Segmentation", "Running Segmentation...", 10)
        
        if not sem_dir.exists():
            print(f"Error: SEM directory not found: {sem_dir}")
            return

        # Run Otsu Segmentation script
        cmd = [
            "python3", "run_otsu_segmentation.py",
            str(sem_dir),
            str(mask_dir)
        ]
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Segmentation failed: {e}")
            update_status("Error", f"Segmentation failed: {e}")
            return

    # 2. Vectorization Stage
    if args.vectorize:
        print("\n=== Stage 2: Vectorization ===")
        update_status("Vectorization", "Starting Vectorization...", 30)
        
        from vectorize_images import vectorize_dual_layer
        
        if not mask_dir.exists():
            print(f"Error: Mask directory not found: {mask_dir}")
            return
            
        # Run Vectorization on the MASKS, not the SEMs
        vectorize_dual_layer(str(mask_dir), str(vector_dir))
        
    # 3. Stitching Stage
    print("\n=== Stage 3: Stitching ===")
    
    # Determine input files
    # We stitch the SVGs in vector_dir
    input_dir = vector_dir
    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        return
        
    svg_files = sorted(glob.glob(str(input_dir / "*.svg")))
    if not svg_files:
        print(f"Error: No SVG files found in {input_dir}")
        if args.vectorize:
            update_status("Error", "No SVGs generated")
        return
        
    print(f"Found {len(svg_files)} SVG files in {input_dir}")
    
    # Run stitching
    subset_files = svg_files[:args.limit]
    print(f"Stitching {len(subset_files)} files...")
    
    update_status("Stitching", "Initializing Stitcher...", 50)

    # Note: reset status file for stitching watcher
    with open('progress_pipeline.json', 'w') as f:
         json.dump({"stage": "Stitching", "status": "Starting stitching process...", "percent": 50}, f)

    stitcher = SVGVectorStitcher(
        raster_method=args.method,
        layout_mode='grid',
        show_labels=args.show_labels,
        show_borders=args.show_borders
    )
    
    output_path = str(base_dir / f'{output_name}.svg')
    success = stitcher.create_panorama_svg(subset_files, output_path)
    
    if success:
        abs_path = os.path.abspath(output_path)
        if os.path.exists(abs_path):
            file_size = os.path.getsize(abs_path)
            print(f"SUCCESS: Successfully created panorama at {abs_path}")
            print(f"File size: {file_size} bytes")
        else:
            print(f"ERROR: Stitcher reported success but file not found at {abs_path}")
            update_status("Error", "Panorama file missing")
            return
    else:
        print("Failed to create panorama")
        update_status("Error", "Stitching failed")
        return

    # 4. GDS Export Stage
    print("\n=== Stage 4: GDS Export ===")
    update_status("GDS Export", "Converting to GDSII...", 90)
        
    from svg_to_gds import svg_to_gds
    output_gds = output_path.replace('.svg', '.gds')
    try:
        svg_to_gds(output_path, output_gds)
        print(f"GDS Export Successful: {output_gds}")
    except Exception as e:
        print(f"GDS Export Failed: {e}")
        update_status("Error", f"GDS Export Failed: {e}")
        return
    
    update_status("Complete", "Pipeline finished successfully!", 100)
    print("Pipeline Finished.")

if __name__ == "__main__":
    main()
