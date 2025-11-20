import numpy as np
import cv2
from typing import List, Tuple, Optional, Dict
from sklearn.neighbors import NearestNeighbors
from svg_vector_analyzer import SVGPathExtractor

class VectorRefiner:
    """
    Vector coordinates based alignment refinement using Iterative Closest Point (ICP)
    """
    
    def __init__(self, max_iterations: int = 20, tolerance: float = 1e-5):
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.extractor = SVGPathExtractor()
        
    def extract_points(self, svg_path: str, num_samples: int = 1000) -> np.ndarray:
        """Extract and sample points from SVG paths"""
        paths = self.extractor.extract_paths_from_svg(svg_path)
        all_points = []
        
        for path in paths:
            coords = path.get('coords', [])
            if len(coords) > 0:
                all_points.extend(coords)
                
        if not all_points:
            return np.zeros((0, 2))
            
        points = np.array(all_points)
        
        # Random sampling if too many points
        if len(points) > num_samples:
            indices = np.random.choice(len(points), num_samples, replace=False)
            points = points[indices]
            
        return points

    def refine_alignment(self, svg1_path: str, svg2_path: str, initial_H: np.ndarray, max_distance: float = 50.0, translation_only: bool = True) -> np.ndarray:
        """
        Refine the homography matrix using ICP on vector points.
        
        Args:
            svg1_path: Source SVG file path
            svg2_path: Target SVG file path
            initial_H: Initial 3x3 homography matrix
            max_distance: Maximum distance for matching points (pixels)
            translation_only: If True, only refine translation (dx, dy)
            
        Returns:
            Refined 3x3 homography matrix
        """
        # 1. Extract points
        src_points = self.extract_points(svg1_path)
        dst_points = self.extract_points(svg2_path)
        
        if len(src_points) < 10 or len(dst_points) < 10:
            print("Warning: Not enough vector points for refinement.")
            return initial_H
            
        # 2. Apply initial transformation to source points
        src_points_h = np.hstack([src_points, np.ones((len(src_points), 1))])
        transformed_src_h = (initial_H @ src_points_h.T).T
        transformed_src = transformed_src_h[:, :2] / transformed_src_h[:, 2:]
        
        # 3. ICP Loop
        refined_H = initial_H.copy()
        
        # Fit NearestNeighbors for dst_points once
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(dst_points)
        
        prev_error = float('inf')
        
        for i in range(self.max_iterations):
            # Find nearest neighbors
            distances, indices = nbrs.kneighbors(transformed_src)
            distances = distances.ravel()
            indices = indices.ravel()
            
            # Filter outliers
            median_dist = np.median(distances)
            valid_mask = (distances < max_distance) & (distances < (median_dist * 3.0))
            
            if np.sum(valid_mask) < 10:
                break
                
            src_match = transformed_src[valid_mask]
            dst_match = dst_points[indices[valid_mask]]
            
            try:
                if translation_only:
                    # Estimate translation only
                    # mean(dst) - mean(src)
                    translation = np.mean(dst_match - src_match, axis=0)
                    dx, dy = translation
                    
                    delta_H = np.eye(3)
                    delta_H[0, 2] = dx
                    delta_H[1, 2] = dy
                else:
                    # Estimate affine transform
                    delta_H_2x3, _ = cv2.estimateAffinePartial2D(src_match, dst_match)
                    if delta_H_2x3 is None:
                        break
                    delta_H = np.eye(3)
                    delta_H[:2, :] = delta_H_2x3
                
                # Update points for next iteration
                src_match_h = np.hstack([transformed_src, np.ones((len(transformed_src), 1))])
                transformed_src_h = (delta_H @ src_match_h.T).T
                transformed_src = transformed_src_h[:, :2] 
                
                # Accumulate total transformation
                refined_H = delta_H @ refined_H
                
                # Check convergence
                mean_error = np.mean(distances[valid_mask])
                if abs(prev_error - mean_error) < self.tolerance:
                    break
                prev_error = mean_error
                
            except Exception as e:
                print(f"ICP Warning: {e}")
                break
        
        # Sanity Check: Compare refined_H with initial_H
        # Extract translation difference
        diff_x = abs(refined_H[0, 2] - initial_H[0, 2])
        diff_y = abs(refined_H[1, 2] - initial_H[1, 2])
        
        # Extract scale difference (approximate from determinant)
        det_initial = np.linalg.det(initial_H[:2, :2])
        det_refined = np.linalg.det(refined_H[:2, :2])
        scale_change = abs(det_refined / (det_initial + 1e-6))
        
        if diff_x > 2000 or diff_y > 2000:  # Allow some shift but not explosion
            print(f"  Warning: Refinement rejected due to large shift (dx={diff_x:.1f}, dy={diff_y:.1f})")
            return initial_H
            
        if scale_change < 0.5 or scale_change > 2.0:
            print(f"  Warning: Refinement rejected due to large scale change ({scale_change:.2f})")
            return initial_H
            
        return refined_H
