"""
이미지 정렬 및 호모그래피 계산 모듈
"""
import numpy as np
import cv2
from typing import Dict, Tuple, Optional, List
from scipy.spatial.distance import cdist


class ImageAligner:
    """이미지 정렬 및 변환 행렬 계산 클래스"""
    
    def __init__(self, method: str = 'homography', ransac_threshold: float = 5.0):
        """
        Args:
            method: 'homography' 또는 'affine'
            ransac_threshold: RANSAC 임계값 (픽셀)
        """
        self.method = method
        self.ransac_threshold = ransac_threshold
    
    def compute_homography(self, matches: Dict) -> Optional[np.ndarray]:
        """
        호모그래피 행렬 계산
        
        Args:
            matches: feature_matcher에서 반환된 매칭 결과
            
        Returns:
            3x3 호모그래피 행렬 또는 None
        """
        pts1 = matches['keypoints0']
        pts2 = matches['keypoints1']
        
        if len(pts1) < 4:
            return None
        
        if self.method == 'homography':
            # 호모그래피 계산 (위치 계산용, 경로 변환에는 사용하지 않음)
            H, mask = cv2.findHomography(
                pts1, pts2,
                method=cv2.RANSAC,
                ransacReprojThreshold=self.ransac_threshold,
                confidence=0.99,
                maxIters=2000
            )
        elif self.method == 'affine':
            H, mask = cv2.estimateAffinePartial2D(
                pts1, pts2,
                method=cv2.RANSAC,
                ransacReprojThreshold=self.ransac_threshold,
                confidence=0.99,
                maxIters=2000
            )
            # Affine을 Homography 형태로 변환
            if H is not None:
                H_3x3 = np.eye(3)
                H_3x3[:2, :] = H
                H = H_3x3
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        return H
    
    def estimate_overlap(self, img1_shape: Tuple[int, int], 
                        img2_shape: Tuple[int, int], 
                        H: np.ndarray) -> float:
        """
        두 이미지 간 겹침 영역 추정
        
        Args:
            img1_shape: 첫 번째 이미지 크기 (H, W)
            img2_shape: 두 번째 이미지 크기 (H, W)
            H: 호모그래피 행렬
            
        Returns:
            겹침 비율 (0.0 ~ 1.0)
        """
        h1, w1 = img1_shape[:2]
        h2, w2 = img2_shape[:2]
        
        # img1의 네 모서리
        corners1 = np.float32([
            [0, 0], [w1, 0], [w1, h1], [0, h1]
        ])
        
        # img1의 모서리를 img2 좌표계로 변환
        corners1_transformed = cv2.perspectiveTransform(
            corners1.reshape(1, -1, 2), H
        ).reshape(-1, 2)
        
        # img2의 경계
        x_min, x_max = 0, w2
        y_min, y_max = 0, h2
        
        # 변환된 모서리 중 img2 경계 내에 있는 점 수
        inside = np.sum(
            (corners1_transformed[:, 0] >= x_min) &
            (corners1_transformed[:, 0] <= x_max) &
            (corners1_transformed[:, 1] >= y_min) &
            (corners1_transformed[:, 1] <= y_max)
        )
        
        # 겹침 비율 (간단한 추정)
        overlap_ratio = inside / 4.0
        
        return overlap_ratio
    
    def find_image_order(self, images: list, matcher, aligner) -> List[int]:
        """
        이미지들의 올바른 순서 찾기 (그래프 기반)
        
        Args:
            images: 이미지 리스트
            matcher: 특징점 매칭 객체
            aligner: 이미지 정렬 객체
            
        Returns:
            정렬된 이미지 인덱스 리스트
        """
        n = len(images)
        if n <= 1:
            return list(range(n))
        
        # 인접 행렬 생성 (매칭 점 수 기반)
        adjacency_matrix = np.zeros((n, n))
        homographies = {}
        
        for i in range(n):
            for j in range(i + 1, n):
                matches = matcher.match_features(images[i], images[j])
                if matches['num_matches'] > 10:
                    H = aligner.compute_homography(matches)
                    if H is not None:
                        overlap = aligner.estimate_overlap(
                            images[i].shape, images[j].shape, H
                        )
                        if overlap > 0.1:  # 최소 10% 겹침
                            adjacency_matrix[i, j] = matches['num_matches']
                            adjacency_matrix[j, i] = matches['num_matches']
                            homographies[(i, j)] = H
                            homographies[(j, i)] = np.linalg.inv(H)
        
        # 간단한 그리디 접근: 가장 많은 연결을 가진 이미지부터 시작
        visited = [False] * n
        order = []
        
        # 시작점 찾기 (가장 적은 연결을 가진 이미지가 끝점일 가능성)
        degrees = np.sum(adjacency_matrix > 0, axis=1)
        start_idx = np.argmin(degrees) if np.any(degrees > 0) else 0
        
        # DFS로 순서 찾기
        def dfs(node):
            visited[node] = True
            order.append(node)
            
            # 인접 노드 중 방문하지 않은 것 찾기
            neighbors = np.where(adjacency_matrix[node] > 0)[0]
            neighbors = [n for n in neighbors if not visited[n]]
            
            if neighbors:
                # 가장 많은 매칭을 가진 이웃 선택
                best_neighbor = max(neighbors, 
                                  key=lambda x: adjacency_matrix[node, x])
                dfs(best_neighbor)
        
        dfs(start_idx)
        
        # 방문하지 않은 노드 추가
        for i in range(n):
            if not visited[i]:
                order.append(i)
        
        return order

