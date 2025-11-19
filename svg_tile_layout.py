"""
2D 타일 레이아웃 계산 모듈
이미지들의 상대적 위치를 분석하여 그리드 형태로 배치
"""
import numpy as np
from typing import List, Dict, Tuple, Optional
from svg_vector_analyzer import VectorFeatureMatcher, SVGPathExtractor
from image_aligner import ImageAligner


class TileLayoutCalculator:
    """2D 타일 레이아웃 계산 클래스"""
    
    def __init__(self):
        self.matcher = VectorFeatureMatcher(use_transformer=False)
        self.aligner = ImageAligner()
        self.extractor = SVGPathExtractor()
    
    def analyze_relative_position(self, svg1_path: str, svg2_path: str, 
                                 H: np.ndarray) -> Dict[str, float]:
        """
        두 이미지 간 상대적 위치 분석
        
        Args:
            svg1_path: 첫 번째 SVG
            svg2_path: 두 번째 SVG
            H: 호모그래피 행렬 (svg1 -> svg2)
            
        Returns:
            위치 정보 딕셔너리
        """
        # SVG 크기 가져오기
        feat1 = self.matcher.extract_features(svg1_path)
        feat2 = self.matcher.extract_features(svg2_path)
        
        if feat1['bbox'] is None or feat2['bbox'] is None:
            return {'direction': 'unknown', 'offset_x': 0, 'offset_y': 0}
        
        w1 = feat1['bbox']['max_x'] - feat1['bbox']['min_x']
        h1 = feat1['bbox']['max_y'] - feat1['bbox']['min_y']
        w2 = feat2['bbox']['max_x'] - feat2['bbox']['min_x']
        h2 = feat2['bbox']['max_y'] - feat2['bbox']['min_y']
        
        # svg1의 중심점을 svg2 좌표계로 변환
        center1 = np.array([[w1/2, h1/2]], dtype=np.float32)
        center1_homogeneous = np.column_stack([center1, np.ones(1)])
        center1_in_svg2 = (H @ center1_homogeneous.T).T
        center1_in_svg2 = center1_in_svg2[:, :2] / center1_in_svg2[:, 2:3]
        
        # svg2의 중심점
        center2 = np.array([[w2/2, h2/2]], dtype=np.float32)
        
        # 상대적 위치 계산
        offset_x = center1_in_svg2[0, 0] - center2[0, 0]
        offset_y = center1_in_svg2[0, 1] - center2[0, 1]
        
        # 방향 결정
        threshold = min(w1, w2, h1, h2) * 0.3
        
        if abs(offset_x) > abs(offset_y):
            if offset_x > threshold:
                direction = 'right'
            elif offset_x < -threshold:
                direction = 'left'
            else:
                direction = 'overlap'
        else:
            if offset_y > threshold:
                direction = 'down'
            elif offset_y < -threshold:
                direction = 'up'
            else:
                direction = 'overlap'
        
        return {
            'direction': direction,
            'offset_x': offset_x,
            'offset_y': offset_y
        }
    
    def compute_tile_positions(self, svg_files: List[str]) -> Dict[int, Tuple[int, int]]:
        """
        이미지들의 2D 타일 위치 계산
        
        Args:
            svg_files: SVG 파일 경로 리스트
            
        Returns:
            {이미지 인덱스: (row, col)} 딕셔너리
        """
        n = len(svg_files)
        if n <= 1:
            return {0: (0, 0)}
        
        # 모든 이미지 쌍 간의 관계 분석
        relationships = {}
        homographies = {}
        
        print("Analyzing image relationships...")
        for i in range(n):
            for j in range(i + 1, n):
                matches = self.matcher.match_vector_features(svg_files[i], svg_files[j])
                if matches['num_matches'] >= 4:
                    H = self.aligner.compute_homography(matches)
                    if H is not None:
                        pos_info = self.analyze_relative_position(
                            svg_files[i], svg_files[j], H
                        )
                        relationships[(i, j)] = pos_info
                        relationships[(j, i)] = {
                            'direction': self._reverse_direction(pos_info['direction']),
                            'offset_x': -pos_info['offset_x'],
                            'offset_y': -pos_info['offset_y']
                        }
                        homographies[(i, j)] = H
                        homographies[(j, i)] = np.linalg.inv(H)
        
        # 첫 번째 이미지를 (0, 0)에 배치
        positions = {0: (0, 0)}
        visited = {0}
        
        # BFS로 위치 계산 (개선된 버전)
        queue = [0]
        position_conflicts = {}  # 위치 충돌 추적
        
        while queue:
            current = queue.pop(0)
            current_pos = positions[current]
            
            # 인접한 이미지 찾기
            neighbors = []
            for (i, j), rel_info in relationships.items():
                if i == current and j not in visited:
                    neighbors.append((j, rel_info))
            
            # 방향별로 정렬 (우선순위: right > down > left > up)
            direction_priority = {'right': 0, 'down': 1, 'left': 2, 'up': 3}
            neighbors.sort(key=lambda x: direction_priority.get(x[1]['direction'], 4))
            
            for j, rel_info in neighbors:
                direction = rel_info['direction']
                
                # 방향에 따라 위치 계산
                if direction == 'right':
                    new_pos = (current_pos[0], current_pos[1] + 1)
                elif direction == 'left':
                    new_pos = (current_pos[0], current_pos[1] - 1)
                elif direction == 'down':
                    new_pos = (current_pos[0] + 1, current_pos[1])
                elif direction == 'up':
                    new_pos = (current_pos[0] - 1, current_pos[1])
                else:  # overlap or unknown
                    # 겹치는 경우 오른쪽에 배치
                    new_pos = (current_pos[0], current_pos[1] + 1)
                
                # 위치 충돌 확인 및 해결
                if new_pos in positions.values():
                    # 충돌 시 다른 위치 찾기
                    row, col = new_pos
                    # 주변 위치 시도
                    for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, 1), (1, -1), (-1, -1)]:
                        candidate_pos = (row + dr, col + dc)
                        if candidate_pos not in positions.values():
                            new_pos = candidate_pos
                            break
                    else:
                        # 모든 주변 위치가 차있으면 순차적으로 배치
                        max_col = max([p[1] for p in positions.values()]) if positions else 0
                        new_pos = (row, max_col + 1)
                
                positions[j] = new_pos
                visited.add(j)
                queue.append(j)
        
        # 모든 이미지를 정사각형 그리드로 배치
        import math
        grid_size = int(math.ceil(math.sqrt(n)))
        
        # 모든 이미지를 정사각형 그리드에 배치
        new_positions = {}
        for idx in range(n):
            row = idx // grid_size
            col = idx % grid_size
            new_positions[idx] = (row, col)
        
        return new_positions
    
    def _reverse_direction(self, direction: str) -> str:
        """방향 반전"""
        mapping = {
            'right': 'left',
            'left': 'right',
            'up': 'down',
            'down': 'up',
            'overlap': 'overlap',
            'unknown': 'unknown'
        }
        return mapping.get(direction, 'unknown')
    
    def calculate_grid_size(self, positions: Dict[int, Tuple[int, int]]) -> Tuple[int, int, int, int]:
        """
        그리드 크기 계산
        
        Returns:
            (min_row, max_row, min_col, max_col)
        """
        if not positions:
            return (0, 0, 0, 0)
        
        rows = [pos[0] for pos in positions.values()]
        cols = [pos[1] for pos in positions.values()]
        
        return (min(rows), max(rows), min(cols), max(cols))
    
    def normalize_positions(self, positions: Dict[int, Tuple[int, int]]) -> Dict[int, Tuple[int, int]]:
        """
        위치를 0부터 시작하도록 정규화
        
        Returns:
            정규화된 위치 딕셔너리
        """
        if not positions:
            return positions
        
        min_row, max_row, min_col, max_col = self.calculate_grid_size(positions)
        
        normalized = {}
        for idx, (row, col) in positions.items():
            normalized[idx] = (row - min_row, col - min_col)
        
        return normalized

