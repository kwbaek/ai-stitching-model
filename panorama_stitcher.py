"""
파노라마 스티칭 메인 파이프라인
"""
import numpy as np
import cv2
from typing import List, Tuple, Optional
from pathlib import Path
import torch

from svg_converter import SVGConverter
from feature_matcher import DeepFeatureMatcher, TraditionalFeatureMatcher
from image_aligner import ImageAligner


class PanoramaStitcher:
    """파노라마 스티칭 메인 클래스"""
    
    def __init__(self, 
                 feature_method: str = 'loftr',
                 use_deep_learning: bool = True,
                 output_size: Tuple[int, int] = (2048, 1768)):
        """
        Args:
            feature_method: 'loftr', 'disk', 'sift', 'orb'
            use_deep_learning: 딥러닝 모델 사용 여부
            output_size: SVG 변환 크기
        """
        self.output_size = output_size
        self.converter = SVGConverter(output_size=output_size)
        
        # 특징점 매칭 모델 선택
        if use_deep_learning and feature_method in ['loftr', 'disk']:
            self.matcher = DeepFeatureMatcher(method=feature_method)
        else:
            self.matcher = TraditionalFeatureMatcher(method=feature_method)
        
        self.aligner = ImageAligner()
    
    def blend_images(self, img1: np.ndarray, img2: np.ndarray, 
                    H: np.ndarray, blend_method: str = 'linear') -> np.ndarray:
        """
        두 이미지를 블렌딩하여 합성
        
        Args:
            img1: 첫 번째 이미지
            img2: 두 번째 이미지
            H: 호모그래피 행렬
            blend_method: 'linear', 'multiband', 'none'
            
        Returns:
            합성된 이미지
        """
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        # img1을 img2 좌표계로 변환
        corners1 = np.float32([
            [0, 0], [w1, 0], [w1, h1], [0, h1]
        ])
        corners1_transformed = cv2.perspectiveTransform(
            corners1.reshape(1, -1, 2), H
        ).reshape(-1, 2)
        
        # 전체 캔버스 크기 계산
        all_corners = np.vstack([
            corners1_transformed,
            np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]])
        ])
        
        x_min, y_min = np.int32(all_corners.min(axis=0))
        x_max, y_max = np.int32(all_corners.max(axis=0))
        
        # 오프셋 계산
        offset_x = -x_min
        offset_y = -y_min
        canvas_w = x_max - x_min
        canvas_h = y_max - y_min
        
        # 변환 행렬에 오프셋 추가
        M = np.array([
            [1, 0, offset_x],
            [0, 1, offset_y],
            [0, 0, 1]
        ]) @ H
        
        # 이미지 변환
        img1_warped = cv2.warpPerspective(
            img1, M, (canvas_w, canvas_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )
        
        img2_placed = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        img2_placed[offset_y:offset_y+h2, offset_x:offset_x+w2] = img2
        
        # 마스크 생성
        mask1 = (img1_warped.sum(axis=2) > 0).astype(np.float32)
        mask2 = (img2_placed.sum(axis=2) > 0).astype(np.float32)
        
        # 겹침 영역
        overlap = mask1 * mask2
        
        if blend_method == 'linear':
            # 선형 블렌딩
            alpha = 0.5
            blended = img1_warped.copy().astype(np.float32)
            
            # 겹침 영역 블렌딩
            blended[overlap > 0] = (
                alpha * img1_warped[overlap > 0] +
                (1 - alpha) * img2_placed[overlap > 0]
            )
            
            # img2만 있는 영역
            blended[(mask2 > 0) & (mask1 == 0)] = img2_placed[(mask2 > 0) & (mask1 == 0)]
            
        elif blend_method == 'multiband':
            # 멀티밴드 블렌딩 (간단한 버전)
            # 거리 기반 가중치
            dist1 = cv2.distanceTransform(
                (mask1 > 0).astype(np.uint8), 
                cv2.DIST_L2, 5
            )
            dist2 = cv2.distanceTransform(
                (mask2 > 0).astype(np.uint8), 
                cv2.DIST_L2, 5
            )
            
            # 가중치 정규화
            total_dist = dist1 + dist2 + 1e-10
            w1 = dist1 / total_dist
            w2 = dist2 / total_dist
            
            blended = (
                w1[:, :, np.newaxis] * img1_warped.astype(np.float32) +
                w2[:, :, np.newaxis] * img2_placed.astype(np.float32)
            )
        else:
            # 블렌딩 없음 (img2 우선)
            blended = img1_warped.copy().astype(np.float32)
            blended[mask2 > 0] = img2_placed[mask2 > 0]
        
        return np.clip(blended, 0, 255).astype(np.uint8)
    
    def stitch_pair(self, img1: np.ndarray, img2: np.ndarray) -> Optional[np.ndarray]:
        """두 이미지를 스티칭"""
        matches = self.matcher.match_features(img1, img2)
        
        if matches['num_matches'] < 10:
            print(f"Not enough matches: {matches['num_matches']}")
            return None
        
        H = self.aligner.compute_homography(matches)
        
        if H is None:
            print("Failed to compute homography")
            return None
        
        result = self.blend_images(img1, img2, H)
        return result
    
    def stitch_sequence(self, images: List[np.ndarray], 
                      order: Optional[List[int]] = None) -> np.ndarray:
        """
        이미지 시퀀스를 순차적으로 스티칭
        
        Args:
            images: 이미지 리스트
            order: 이미지 순서 (None이면 자동 탐지)
            
        Returns:
            최종 파노라마 이미지
        """
        if len(images) == 0:
            raise ValueError("No images provided")
        
        if len(images) == 1:
            return images[0]
        
        # 순서 자동 탐지
        if order is None:
            print("Finding image order...")
            order = self.aligner.find_image_order(images, self.matcher, self.aligner)
            print(f"Image order: {order}")
        
        # 순서대로 이미지 재배열
        ordered_images = [images[i] for i in order]
        
        # 순차적으로 스티칭
        result = ordered_images[0]
        
        for i in range(1, len(ordered_images)):
            print(f"Stitching image {i+1}/{len(ordered_images)}...")
            new_result = self.stitch_pair(result, ordered_images[i])
            
            if new_result is not None:
                result = new_result
            else:
                print(f"Warning: Failed to stitch image {i+1}, skipping...")
        
        return result
    
    def stitch_from_svg_dir(self, svg_dir: str, 
                           output_path: str,
                           max_images: Optional[int] = None) -> np.ndarray:
        """
        SVG 디렉토리에서 파노라마 생성
        
        Args:
            svg_dir: SVG 파일 디렉토리
            output_path: 출력 파일 경로
            max_images: 최대 이미지 수 (None이면 모두 사용)
            
        Returns:
            최종 파노라마 이미지
        """
        print(f"Converting SVG files from {svg_dir}...")
        images = self.converter.convert_directory(svg_dir)
        
        if len(images) == 0:
            raise ValueError("No images converted")
        
        if max_images:
            images = images[:max_images]
        
        print(f"Stitching {len(images)} images...")
        panorama = self.stitch_sequence(images)
        
        # 결과 저장
        cv2.imwrite(output_path, cv2.cvtColor(panorama, cv2.COLOR_RGB2BGR))
        print(f"Panorama saved to {output_path}")
        
        return panorama



