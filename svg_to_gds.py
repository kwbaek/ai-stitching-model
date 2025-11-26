import gdstk
import xml.etree.ElementTree as ET
from svg.path import parse_path
import numpy as np
import sys
import os

def svg_to_gds(svg_file, gds_file, layer_map=None):
    """
    Convert an SVG file to a GDSII file.
    
    Args:
        svg_file (str): Path to the input SVG file.
        gds_file (str): Path to the output GDSII file.
        layer_map (dict): Optional mapping of SVG styles/IDs to GDS layers.
    """
    print(f"Converting {svg_file} to {gds_file}...")
    
    # Parse SVG
    tree = ET.parse(svg_file)
    root = tree.getroot()
    
    # Get SVG dimensions
    width_str = root.get('width', '0').replace('px', '')
    height_str = root.get('height', '0').replace('px', '')
    
    try:
        svg_width = float(width_str)
        svg_height = float(height_str)
    except ValueError:
        # Try to parse viewBox if width/height are missing or invalid
        viewbox = root.get('viewBox')
        if viewbox:
            _, _, w, h = map(float, viewbox.split())
            svg_width = w
            svg_height = h
        else:
            print("Warning: Could not determine SVG dimensions. Defaulting to 1000x1000.")
            svg_width = 1000.0
            svg_height = 1000.0
            
    print(f"SVG Dimensions: {svg_width} x {svg_height}")

    # Create GDSII Library and Cell
    lib = gdstk.Library()
    cell = lib.new_cell('PANORAMA')
    
    # Namespace for SVG parsing (often needed)
    ns = {'svg': 'http://www.w3.org/2000/svg'}
    
    # Helper to flip Y coordinate (SVG is Y-down, GDS is Y-up)
    def flip_y(y):
        return svg_height - y

    element_counts = {'path': 0, 'rect': 0, 'circle': 0, 'text': 0, 'other': 0}

    # Iterate over all elements
    # Note: This is a flat iteration. Groups (<g>) are ignored for hierarchy but their children are processed if we use .iter()
    
    # 1. Process Paths
    for path in root.iter('{http://www.w3.org/2000/svg}path'):
        try:
            d = path.get('d')
            if not d:
                continue
                
            parsed_path = parse_path(d)
            points = []
            
            # Sample points along the path
            # For straight lines, start/end is enough, but for curves we need sampling.
            # A simple approach is to sample at fixed resolution or use the segment points.
            
            # We'll sample 10 points per segment for curves, 2 for lines
            for segment in parsed_path:
                num_samples = 2
                if segment.__class__.__name__ in ['CubicBezier', 'QuadraticBezier', 'Arc']:
                    num_samples = 10
                
                for i in range(num_samples):
                    t = i / (num_samples - 1) if num_samples > 1 else 0
                    pt = segment.point(t)
                    points.append((pt.real, flip_y(pt.imag)))
            
            if points:
                # Create polygon (layer 1 for paths)
                poly = gdstk.Polygon(points, layer=1)
                cell.add(poly)
                element_counts['path'] += 1
                
        except Exception as e:
            print(f"Error processing path: {e}")

    # 2. Process Rects
    for rect in root.iter('{http://www.w3.org/2000/svg}rect'):
        try:
            x = float(rect.get('x', 0))
            y = float(rect.get('y', 0))
            w = float(rect.get('width', 0))
            h = float(rect.get('height', 0))
            
            # Skip if border (often used for page bounds) or very large
            # But user wanted borders, so we keep them.
            # Layer 2 for rects
            
            # GDS Rectangle: (x1, y1), (x2, y2)
            # SVG y is top-left. GDS y is bottom-left.
            # SVG Rect: (x, y) is top-left.
            # y_gds_top = flip_y(y)
            # y_gds_bottom = flip_y(y + h)
            
            p1 = (x, flip_y(y + h))
            p2 = (x + w, flip_y(y))
            
            rectangle = gdstk.rectangle(p1, p2, layer=2)
            cell.add(rectangle)
            element_counts['rect'] += 1
            
        except Exception as e:
            print(f"Error processing rect: {e}")

    # 3. Process Circles
    for circle in root.iter('{http://www.w3.org/2000/svg}circle'):
        try:
            cx = float(circle.get('cx', 0))
            cy = float(circle.get('cy', 0))
            r = float(circle.get('r', 0))
            
            # Layer 3 for circles
            center = (cx, flip_y(cy))
            circle_poly = gdstk.ellipse(center, r, layer=3, tolerance=0.1)
            cell.add(circle_poly)
            element_counts['circle'] += 1
            
        except Exception as e:
            print(f"Error processing circle: {e}")

    # 4. Process Text
    for text in root.iter('{http://www.w3.org/2000/svg}text'):
        try:
            x = float(text.get('x', 0))
            y = float(text.get('y', 0))
            content = text.text
            
            if content:
                # Layer 4 for text
                # GDS labels are just points with text attached
                label = gdstk.Label(content, (x, flip_y(y)), layer=4)
                cell.add(label)
                element_counts['text'] += 1
                
        except Exception as e:
            print(f"Error processing text: {e}")

    # Save GDS
    lib.write_gds(gds_file)
    
    print(f"Conversion complete!")
    print(f"Stats:")
    for k, v in element_counts.items():
        print(f"  {k}: {v}")
    print(f"Saved to {gds_file}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python svg_to_gds.py <input_svg> [output_gds]")
        sys.exit(1)
        
    input_svg = sys.argv[1]
    if len(sys.argv) >= 3:
        output_gds = sys.argv[2]
    else:
        output_gds = os.path.splitext(input_svg)[0] + ".gds"
        
    svg_to_gds(input_svg, output_gds)
