"""
SVG 벡터 기반 겹침 감지 모듈
두 SVG 파일 간의 겹침 영역을 벡터 좌표로 분석
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from svg_vector_analyzer import SVGPathExtractor


class OverlapDetector:
    """벡터 기반 겹침 감지 클래스"""
    
    def __init__(self, overlap_threshold: float = 0.1):
        """
        Args:
            overlap_threshold: 겹침으로 간주할 최소 비율 (0.0 ~ 1.0)
        """
        self.extractor = SVGPathExtractor()
        self.overlap_threshold = overlap_threshold
    
    def get_bounding_box(self, svg_path: str) -> Optional[Dict[str, float]]:
        """
        SVG 파일의 바운딩 박스 계산
        
        Args:
            svg_path: SVG 파일 경로
            
        Returns:
            바운딩 박스 {'min_x', 'max_x', 'min_y', 'max_y', 'width', 'height'}
        """
        paths = self.extractor.extract_paths_from_svg(svg_path)
        
        if not paths:
            return None
        
        all_coords = []
        for path_info in paths:
            coords = path_info.get('coords', [])
            if coords:
                all_coords.extend(coords)
        
        if not all_coords:
            return None
        
        coords_array = np.array(all_coords)
        min_x, min_y = coords_array.min(axis=0)
        max_x, max_y = coords_array.max(axis=0)
        
        return {
            'min_x': float(min_x),
            'max_x': float(max_x),
            'min_y': float(min_y),
            'max_y': float(max_y),
            'width': float(max_x - min_x),
            'height': float(max_y - min_y)
        }
    
    def compute_bbox_intersection(self, bbox1: Dict, bbox2: Dict) -> Optional[Dict]:
        """
        두 바운딩 박스의 교차 영역 계산
        
        Args:
            bbox1: 첫 번째 바운딩 박스
            bbox2: 두 번째 바운딩 박스
            
        Returns:
            교차 영역 또는 None
        """
        # 교차 영역 계산
        inter_min_x = max(bbox1['min_x'], bbox2['min_x'])
        inter_max_x = min(bbox1['max_x'], bbox2['max_x'])
        inter_min_y = max(bbox1['min_y'], bbox2['min_y'])
        inter_max_y = min(bbox1['max_y'], bbox2['max_y'])
        
        # 교차하지 않는 경우
        if inter_min_x >= inter_max_x or inter_min_y >= inter_max_y:
            return None
        
        return {
            'min_x': inter_min_x,
            'max_x': inter_max_x,
            'min_y': inter_min_y,
            'max_y': inter_max_y,
            'width': inter_max_x - inter_min_x,
            'height': inter_max_y - inter_min_y
        }
    
    def compute_overlap_score(self, bbox1: Dict, bbox2: Dict, 
                             intersection: Dict) -> float:
        """
        겹침 정도 점수 계산 (0.0 ~ 1.0)
        
        Args:
            bbox1: 첫 번째 바운딩 박스
            bbox2: 두 번째 바운딩 박스
            intersection: 교차 영역
            
        Returns:
            겹침 점수 (교차 영역 / 작은 박스 면적)
        """
        if intersection is None:
            return 0.0
        
        area1 = bbox1['width'] * bbox1['height']
        area2 = bbox2['width'] * bbox2['height']
        inter_area = intersection['width'] * intersection['height']
        
        # 작은 박스 대비 겹침 비율
        min_area = min(area1, area2)
        if min_area == 0:
            return 0.0
        
        return inter_area / min_area
    
    def get_overlap_direction(self, bbox1: Dict, bbox2: Dict, 
                             intersection: Dict) -> str:
        """
        겹침 방향 감지 (left/right/top/bottom)
        
        Args:
            bbox1: 첫 번째 바운딩 박스 (기준)
            bbox2: 두 번째 바운딩 박스
            intersection: 교차 영역
            
        Returns:
            방향 문자열 ('right', 'left', 'bottom', 'top', 'center')
        """
        if intersection is None:
            return 'none'
        
        # bbox1의 중심
        center1_x = (bbox1['min_x'] + bbox1['max_x']) / 2
        center1_y = (bbox1['min_y'] + bbox1['max_y']) / 2
        
        # bbox2의 중심
        center2_x = (bbox2['min_x'] + bbox2['max_x']) / 2
        center2_y = (bbox2['min_y'] + bbox2['max_y']) / 2
        
        # 중심 간 거리
        dx = center2_x - center1_x
        dy = center2_y - center1_y
        
        # 수평/수직 판단
        if abs(dx) > abs(dy):
            # 수평 방향이 더 큼
            return 'right' if dx > 0 else 'left'
        else:
            # 수직 방향이 더 큼
            return 'bottom' if dy > 0 else 'top'
    
    def extract_overlap_region(self, svg_path: str, 
                               region: Dict) -> List[np.ndarray]:
        """
        겹침 영역 내의 벡터 좌표 추출
        
        Args:
            svg_path: SVG 파일 경로
            region: 영역 {'min_x', 'max_x', 'min_y', 'max_y'}
            
        Returns:
            영역 내 좌표 리스트
        """
        paths = self.extractor.extract_paths_from_svg(svg_path)
        
        overlap_coords = []
        for path_info in paths:
            coords = path_info.get('coords', [])
            if not coords:
                continue
            
            coords_array = np.array(coords)
            
            # 영역 내 좌표 필터링
            mask = (
                (coords_array[:, 0] >= region['min_x']) &
                (coords_array[:, 0] <= region['max_x']) &
                (coords_array[:, 1] >= region['min_y']) &
                (coords_array[:, 1] <= region['max_y'])
            )
            
            filtered = coords_array[mask]
            if len(filtered) > 0:
                overlap_coords.append(filtered)
        
        return overlap_coords
    
    def detect_overlap(self, svg1_path: str, svg2_path: str) -> Dict:
        """
        두 SVG 파일 간 겹침 감지
        
        Args:
            svg1_path: 첫 번째 SVG 파일
            svg2_path: 두 번째 SVG 파일
            
        Returns:
            겹침 정보 딕셔너리
        """
        # 바운딩 박스 계산
        bbox1 = self.get_bounding_box(svg1_path)
        bbox2 = self.get_bounding_box(svg2_path)
        
        if bbox1 is None or bbox2 is None:
            return {
                'has_overlap': False,
                'overlap_score': 0.0,
                'direction': 'none',
                'bbox1': bbox1,
                'bbox2': bbox2,
                'intersection': None
            }
        
        # 교차 영역 계산
        intersection = self.compute_bbox_intersection(bbox1, bbox2)
        
        # 겹침 점수 계산
        overlap_score = self.compute_overlap_score(bbox1, bbox2, intersection)
        
        # 방향 감지
        direction = self.get_overlap_direction(bbox1, bbox2, intersection)
        
        # 겹침 여부 판단
        has_overlap = overlap_score >= self.overlap_threshold
        
        result = {
            'has_overlap': has_overlap,
            'overlap_score': overlap_score,
            'direction': direction,
            'bbox1': bbox1,
            'bbox2': bbox2,
            'intersection': intersection
        }
        
        # 겹침 영역 좌표 추출 (옵션)
        if has_overlap and intersection:
            result['overlap_coords1'] = self.extract_overlap_region(svg1_path, intersection)
            result['overlap_coords2'] = self.extract_overlap_region(svg2_path, intersection)
        
        return result
    
    def analyze_sequence_overlaps(self, svg_files: List[str]) -> List[Dict]:
        """
        SVG 파일 시퀀스의 연속적인 겹침 분석
        
        Args:
            svg_files: SVG 파일 경로 리스트
            
        Returns:
            겹침 정보 리스트 [(svg_i, svg_i+1) 간 겹침]
        """
        overlaps = []
        
        for i in range(len(svg_files) - 1):
            overlap_info = self.detect_overlap(svg_files[i], svg_files[i + 1])
            overlap_info['index1'] = i
            overlap_info['index2'] = i + 1
            overlap_info['file1'] = svg_files[i]
            overlap_info['file2'] = svg_files[i + 1]
            overlaps.append(overlap_info)
        
        return overlaps
    
    def find_best_neighbor(self, svg_path: str, 
                          candidate_svgs: List[str]) -> Tuple[Optional[str], float]:
        """
        주어진 SVG와 가장 많이 겹치는 이웃 찾기
        
        Args:
            svg_path: 기준 SVG 파일
            candidate_svgs: 후보 SVG 파일 리스트
            
        Returns:
            (최적 이웃 경로, 겹침 점수)
        """
        best_neighbor = None
        best_score = 0.0
        
        for candidate in candidate_svgs:
            if candidate == svg_path:
                continue
            
            overlap_info = self.detect_overlap(svg_path, candidate)
            
            if overlap_info['overlap_score'] > best_score:
                best_score = overlap_info['overlap_score']
                best_neighbor = candidate
        
        return best_neighbor, best_score
