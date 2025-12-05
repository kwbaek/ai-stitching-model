import os
import sys
import subprocess
import json
import time
from pathlib import Path
from PIL import Image
import multiprocessing

def convert_single_image(args):
    """
    Convert a single image to SVG using potrace.
    Args:
        args: tuple (input_path, output_path, config)
    """
    input_path, output_path, config = args
    
    try:
        # 1. Convert PNG to BMP (potrace needs BMP or PNM)
        # We use BMP as it's simple and lossless
        with Image.open(input_path) as img:
            # Convert to grayscale if not already
            if img.mode != 'L':
                img = img.convert('L')
            
            # Create a temporary BMP file
            bmp_path = str(input_path).replace('.png', '.bmp')
            img.save(bmp_path)
            
        # 2. Run potrace
        # -s: SVG output
        # -k: black level (0.5 default)
        # --flat: don't curve optimization (optional, but we want vectors)
        # Actually, for stitching, we probably want good curves. Default is fine.
        cmd = ['potrace', '-s', bmp_path, '-o', str(output_path)]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # 3. Cleanup BMP
        if os.path.exists(bmp_path):
            os.remove(bmp_path)
            
        return True, str(input_path)
        
    except Exception as e:
        print(f"Error converting {input_path}: {e}")
        return False, str(input_path)

def update_progress(current, total, stage="Vectorization"):
    """Update progress JSON file"""
    progress_data = {
        "stage": stage,
        "current": current,
        "total": total,
        "percent": (current / total) * 100 if total > 0 else 0,
        "status": f"Vectorizing images... ({current}/{total})"
    }
    
    with open('progress_pipeline.json', 'w') as f:
        json.dump(progress_data, f)

def vectorize_images(input_dir, output_dir, num_workers=None):
    """
    Vectorize all PNG images in input_dir to SVG in output_dir.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get list of PNG files
    png_files = sorted(list(input_path.glob('*.png')))
    total_files = len(png_files)
    
    print(f"Found {total_files} PNG files in {input_dir}")
    print(f"Output directory: {output_dir}")
    
    # Prepare arguments for workers
    tasks = []
    for png_file in png_files:
        # Maintain filename but change extension
        svg_file = output_path / png_file.with_suffix('.svg').name
        tasks.append((png_file, svg_file, {}))
    
    # Run in parallel
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()
        
    print(f"Starting vectorization with {num_workers} workers...")
    
    completed = 0
    update_progress(0, total_files)
    
    with multiprocessing.Pool(processes=num_workers) as pool:
        for success, filename in pool.imap_unordered(convert_single_image, tasks):
            completed += 1
            if completed % 10 == 0 or completed == total_files:
                print(f"Progress: {completed}/{total_files}")
                update_progress(completed, total_files)
                
    print("Vectorization complete!")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python vectorize_images.py <input_dir> <output_dir>")
        sys.exit(1)
        
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    
    vectorize_images(input_dir, output_dir)
