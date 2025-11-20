import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from svg_vector_stitcher import SVGVectorStitcher
from feature_matcher import DeepFeatureMatcher

def draw_matches(img1, img2, mkpts0, mkpts1, output_path):
    """Draw matches between two images"""
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # Create a large image to hold both
    vis = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    vis[:h1, :w1] = img1
    vis[:h2, w1:w1+w2] = img2
    
    # Draw lines
    for pt1, pt2 in zip(mkpts0, mkpts1):
        pt1 = (int(pt1[0]), int(pt1[1]))
        pt2 = (int(pt2[0] + w1), int(pt2[1]))
        cv2.line(vis, pt1, pt2, (0, 255, 0), 1)
        cv2.circle(vis, pt1, 2, (0, 0, 255), -1)
        cv2.circle(vis, pt2, 2, (0, 0, 255), -1)
        
    cv2.imwrite(output_path, vis)
    if os.path.exists(output_path):
        print(f"SUCCESS: Saved match visualization to {os.path.abspath(output_path)}")
    else:
        print(f"ERROR: Failed to save match visualization to {output_path}")

def main():
    m2_dir = "/app/data/ai-stitching-model/m2"
    svg1 = os.path.join(m2_dir, "label0000.svg")
    svg2 = os.path.join(m2_dir, "label0001.svg")
    
    print(f"Debugging pair: {os.path.basename(svg1)} <-> {os.path.basename(svg2)}")
    
    # Initialize stitcher to get converter
    stitcher = SVGVectorStitcher(
        use_raster_matching=True,
        raster_method='loftr'
    )
    
    # 1. Convert to image (Low Res)
    print("Converting to images (Low Res)...")
    img1 = stitcher.converter.svg_to_image(svg1)
    img2 = stitcher.converter.svg_to_image(svg2)
    
    # 2. Match
    print("Matching with LoFTR...")
    matcher = DeepFeatureMatcher(method='loftr')
    matches = matcher.match_features(img1, img2)
    
    print(f"Found {matches['num_matches']} matches")
    
    # 3. Visualize
    draw_matches(img1, img2, matches['keypoints0'], matches['keypoints1'], 
                 "/app/data/ai-stitching-model/debug_matches_lowres.png")
    
    # 4. Compute Homography
    if matches['num_matches'] >= 4:
        src_pts = matches['keypoints0']
        dst_pts = matches['keypoints1']
        H_low, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        if H_low is not None:
            # Scale Homography
            scale_x = 4096 / 1024
            scale_y = 3536 / 884
            S = np.diag([scale_x, scale_y, 1.0])
            S_inv = np.diag([1.0/scale_x, 1.0/scale_y, 1.0])
            H = S @ H_low @ S_inv
            
            print("Initial Scaled Homography:")
            print(H)
            
            # Refine with VectorRefiner
            from vector_refinement import VectorRefiner
            refiner = VectorRefiner()
            print("Refining with ICP...")
            H_refined = refiner.refine_alignment(svg1, svg2, H)
            
            print("Refined Homography:")
            print(H_refined)
            
            # Visualize Vector Alignment
            print("Visualizing vector alignment...")
            pts1 = refiner.extract_points(svg1, num_samples=2000)
            pts2 = refiner.extract_points(svg2, num_samples=2000)
            
            # Transform pts1
            pts1_h = np.hstack([pts1, np.ones((len(pts1), 1))])
            
            # 1. Apply Initial H
            pts1_init_h = (H @ pts1_h.T).T
            pts1_init = pts1_init_h[:, :2] / pts1_init_h[:, 2:]
            
            # 2. Apply Refined H
            pts1_ref_h = (H_refined @ pts1_h.T).T
            pts1_ref = pts1_ref_h[:, :2] / pts1_ref_h[:, 2:]
            
            # Plot
            plt.figure(figsize=(12, 6))
            
            # Subplot 1: Initial
            plt.subplot(1, 2, 1)
            plt.scatter(pts2[:, 0], pts2[:, 1], c='blue', s=1, label='Target (SVG2)')
            plt.scatter(pts1_init[:, 0], pts1_init[:, 1], c='red', s=1, alpha=0.5, label='Source (Initial H)')
            plt.title("Initial Alignment (Scaled LoFTR)")
            plt.legend()
            plt.axis('equal')
            
            # Subplot 2: Refined
            plt.subplot(1, 2, 2)
            plt.scatter(pts2[:, 0], pts2[:, 1], c='blue', s=1, label='Target (SVG2)')
            plt.scatter(pts1_ref[:, 0], pts1_ref[:, 1], c='green', s=1, alpha=0.5, label='Source (Refined ICP)')
            plt.title("Refined Alignment (ICP)")
            plt.legend()
            plt.axis('equal')
            
            plt.tight_layout()
            plt.tight_layout()
            plt.savefig("/app/data/ai-stitching-model/debug_vector_alignment.png")
            
            if os.path.exists("/app/data/ai-stitching-model/debug_vector_alignment.png"):
                print(f"SUCCESS: Saved debug_vector_alignment.png at {os.path.abspath('/app/data/ai-stitching-model/debug_vector_alignment.png')}")
            else:
                print("ERROR: Failed to save debug_vector_alignment.png")

    print("\nDEBUGGING COMPLETE")

if __name__ == "__main__":
    main()
