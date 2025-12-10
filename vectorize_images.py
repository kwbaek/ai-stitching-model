import os
import sys
import subprocess
import json
import re
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image
from skimage.measure import label, regionprops
import multiprocessing

# Register namespace
ET.register_namespace('', "http://www.w3.org/2000/svg")

def get_potrace_paths(bmp_path, width, height):
    """
    Run potrace and return list of path elements (ET.Element)
    with baked-in transforms and integer coordinates.
    """
    cmd = ['potrace', '-s', '-a', '0', bmp_path, '-o', '-']
    result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    
    svg_content = result.stdout.decode('utf-8')
    try:
        # Parse output
        # We need to wrap it in a root if not already valid XML (it should be)
        root = ET.fromstring(svg_content)
        
        # Find group with transform
        g = root.find('.//{http://www.w3.org/2000/svg}g')
        paths = []
        
        if g is not None:
            transform = g.get('transform', '')
            svg_paths = g.findall('{http://www.w3.org/2000/svg}path')
            
            # Extract potrace transform
            # Usually: translate(0, height) scale(0.1, -0.1)
            tx, ty = 0.0, 0.0
            sx, sy = 1.0, 1.0
            
            if 'translate' in transform:
                match = re.search(r'translate\(([^,]+),([^)]+)\)', transform)
                if match:
                    tx = float(match.group(1))
                    ty = float(match.group(2))
            
            if 'scale' in transform:
                match = re.search(r'scale\(([^,]+),([^)]+)\)', transform)
                if match:
                    sx = float(match.group(1))
                    sy = float(match.group(2))
                    
            for p in svg_paths:
                d = p.get('d')
                
                # Transform path data
                parts = re.split('([a-zA-Z])', d)
                new_d = []
                current_cmd = ''
                
                for part in parts:
                    if not part: continue
                    if part.isalpha():
                        current_cmd = part
                        new_d.append(part)
                    else:
                        # Parse numbers
                        nums = [float(x) for x in part.replace(',', ' ').split() if x.strip()]
                        
                        is_relative = current_cmd.islower()
                        trans_nums = []
                        
                        for i in range(0, len(nums), 2):
                            if i+1 < len(nums):
                                x, y = nums[i], nums[i+1]
                                
                                if is_relative:
                                    nx = x * sx
                                    ny = y * sy
                                else:
                                    nx = x * sx + tx
                                    ny = y * sy + ty
                                    
                                trans_nums.append(f"{int(round(nx))},{int(round(ny))}")
                        
                        new_d.append(" ".join(trans_nums))
                
                final_d = "".join(new_d)
                new_p = ET.Element('path')
                new_p.set('d', final_d)
                new_p.set('fill', 'lime')
                paths.append(new_p)
                
        return paths

    except Exception as e:
        print(f"Error parsing potrace output: {e}")
        return []

def convert_single_item(args):
    stem, tracks_path, vias_path, output_path = args
    
    try:
        width, height = 4096, 3536 # Default fallback
        track_elements = []
        via_elements = []
        
        # 1. Process Tracks (Potrace)
        if tracks_path and tracks_path.exists():
            with Image.open(tracks_path) as img:
                width, height = img.size
                
                bmp_path = str(tracks_path).replace('.png', '.bmp')
                img.save(bmp_path)
                
            track_elements = get_potrace_paths(bmp_path, width, height)
            
            if os.path.exists(bmp_path):
                os.remove(bmp_path)

        # 2. Process Vias (Circles)
        if vias_path and vias_path.exists():
            img = np.array(Image.open(vias_path))
            if img.ndim == 3: img = img[...,0] # Ensure grayscale assumption
            
            labeled_img = label(img > 128)
            regions = regionprops(labeled_img)
            
            for region in regions:
                y, x = region.centroid
                r = region.equivalent_diameter_area / 2.0
                
                c = ET.Element('circle')
                c.set('cx', f"{int(round(x))}")
                c.set('cy', f"{int(round(y))}")
                c.set('r', f"{int(round(r))}")
                c.set('fill', 'red')
                via_elements.append(c)

        # 3. Combine into SVG
        root = ET.Element('svg', xmlns="http://www.w3.org/2000/svg")
        root.set('width', str(width))
        root.set('height', str(height))
        root.set('viewBox', f"0 0 {width} {height}")
        
        # Add tracks first, then vias on top?
        for p in track_elements:
            root.append(p)
            
        for c in via_elements:
            root.append(c)
            
        tree = ET.ElementTree(root)
        tree.write(output_path, encoding='UTF-8', xml_declaration=True)
        
        return True, str(stem)

    except Exception as e:
        print(f"Error converting {stem}: {e}")
        return False, str(stem)

def vectorize_dual_layer(input_dir, output_dir, num_workers=None):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Identify unique stems
    files = list(input_path.glob("*.png"))
    stems = set()
    for f in files:
        if f.name.endswith("_tracks.png"):
            stems.add(f.name.replace("_tracks.png", ""))
        elif f.name.endswith("_vias.png"):
            stems.add(f.name.replace("_vias.png", ""))
            
    stems = sorted(list(stems))
    print(f"Found {len(stems)} unique samples in {input_dir}")
    
    tasks = []
    for stem in stems:
        t_path = input_path / f"{stem}_tracks.png"
        v_path = input_path / f"{stem}_vias.png"
        out_path = output_path / f"{stem}.svg"
        tasks.append((stem, t_path, v_path, out_path))
        
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()
        
    print(f"Starting vectorization with {num_workers} workers...")
    
    with multiprocessing.Pool(processes=num_workers) as pool:
        for success, name in pool.imap_unordered(convert_single_item, tasks):
            if not success:
               print(f"Failed to convert {name}")

    print("Vectorization complete!")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python vectorize_images.py <input_dir> <output_dir>")
        sys.exit(1)
        
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    
    vectorize_dual_layer(input_dir, output_dir)
