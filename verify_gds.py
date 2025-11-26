import gdstk
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import numpy as np
import os

def verify_gds(gds_file, output_png):
    print(f"Reading {gds_file}...")
    lib = gdstk.read_gds(gds_file)
    cell = lib.top_level()[0]
    
    print(f"Top cell: {cell.name}")
    print(f"Polygons: {len(cell.polygons)}")
    print(f"Labels: {len(cell.labels)}")
    
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 10))
    
    patches = []
    # Plot a subset of polygons to avoid memory issues if too large
    # But 200k might be okay for a static png if we are careful.
    # Let's plot every 10th polygon to be safe and fast.
    
    print("Generating preview plot...")
    for i, poly in enumerate(cell.polygons):
        if i % 10 == 0: # Downsample
            patches.append(Polygon(poly.points))
            
    p = PatchCollection(patches, alpha=0.5, fc='#00f3ff', ec='none')
    ax.add_collection(p)
    
    # Add labels
    # for label in cell.labels:
    #     ax.text(label.origin[0], label.origin[1], label.text, fontsize=5, color='white')
        
    ax.autoscale()
    ax.set_aspect('equal')
    ax.set_facecolor('#050510') # Dark background
    fig.patch.set_facecolor('#050510')
    
    # Remove axes
    ax.axis('off')
    
    plt.title(f"GDSII Preview: {cell.name} ({len(cell.polygons)} polygons)", color='white')
    plt.savefig(output_png, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved preview to {output_png}")

if __name__ == "__main__":
    verify_gds('panorama_m2.gds', 'gds_verification.png')
