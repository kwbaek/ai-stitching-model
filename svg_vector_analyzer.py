"""
SVG 벡터 데이터 직접 분석 및 매칭 모듈
SVG 경로 좌표를 추출하여 벡터 기반 매칭 수행
"""
import re
import numpy as np
import xml.etree.ElementTree as ET
from typing import List, Dict, Tuple, Optional
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import cv2


class SVGPathExtractor:
    """SVG 경로 데이터 추출 클래스"""
    
    @staticmethod
    def parse_path_d(path_d: str) -> List[Tuple[float, float]]:
        """
        SVG path d 속성에서 좌표 추출
        
        Args:
            path_d: SVG path의 d 속성 값 (예: "M2725,3516 L2710,3518 ...")
            
        Returns:
            좌표 리스트 [(x, y), ...]
        """
        coordinates = []
        current_x, current_y = 0, 0
        
        # 모든 숫자 추출 (음수 포함)
        numbers = re.findall(r'[-]?\d+\.?\d*', path_d)
        if not numbers:
            return coordinates
        
        numbers = [float(n) for n in numbers]
        
        # 명령어 추출
        commands = re.findall(r'[MLHVCSQTAZmlhvcsqtaz]', path_d)
        
        if not commands:
            # 명령어가 없으면 숫자 쌍으로 처리
            if len(numbers) >= 2:
                for i in range(0, len(numbers) - 1, 2):
                    coordinates.append((numbers[i], numbers[i + 1]))
            return coordinates
        
        i = 0
        for cmd in commands:
            cmd_upper = cmd.upper()
            
            if cmd_upper == 'M':  # Move to
                if i < len(numbers) - 1:
                    current_x = numbers[i]
                    current_y = numbers[i + 1]
                    coordinates.append((current_x, current_y))
                    i += 2
            elif cmd_upper == 'L':  # Line to
                if i < len(numbers) - 1:
                    current_x = numbers[i]
                    current_y = numbers[i + 1]
                    coordinates.append((current_x, current_y))
                    i += 2
            elif cmd_upper == 'H':  # Horizontal line
                if i < len(numbers):
                    current_x = numbers[i]
                    coordinates.append((current_x, current_y))
                    i += 1
            elif cmd_upper == 'V':  # Vertical line
                if i < len(numbers):
                    current_y = numbers[i]
                    coordinates.append((current_x, current_y))
                    i += 1
            elif cmd_upper == 'Z':  # Close path
                if len(coordinates) > 0:
                    # 첫 번째 점으로 돌아가기
                    coordinates.append(coordinates[0])
            elif cmd_upper == 'C':  # Cubic bezier
                if i < len(numbers) - 5:
                    end_x = numbers[i + 4]
                    end_y = numbers[i + 5]
                    coordinates.append((end_x, end_y))
                    current_x, current_y = end_x, end_y
                    i += 6
            elif cmd_upper == 'Q':  # Quadratic bezier
                if i < len(numbers) - 3:
                    end_x = numbers[i + 2]
                    end_y = numbers[i + 3]
                    coordinates.append((end_x, end_y))
                    current_x, current_y = end_x, end_y
                    i += 4
        
        return coordinates
    
    @staticmethod
    def extract_paths_from_svg(svg_path: str) -> List[Dict]:
        """
        SVG 파일에서 모든 path 요소 추출
        
        Args:
            svg_path: SVG 파일 경로
            
        Returns:
            path 정보 리스트 [{'d': path_d, 'fill': fill, 'coords': coords}, ...]
        """
        try:
            tree = ET.parse(svg_path)
            root = tree.getroot()
        except Exception as e:
            print(f"Error parsing SVG: {e}")
            return []
        
        # SVG 네임스페이스 처리 (없을 수도 있음)
        paths = []
        
        # 네임스페이스가 있는 경우와 없는 경우 모두 처리
        for path_elem in root.findall('.//{http://www.w3.org/2000/svg}path') + root.findall('.//path'):
            path_d = path_elem.get('d', '')
            fill = path_elem.get('fill', 'none')
            
            if path_d:
                coords = SVGPathExtractor.parse_path_d(path_d)
                if len(coords) > 0:
                    paths.append({
                        'd': path_d,
                        'fill': fill,
                        'coords': coords
                    })
        
        return paths
    
    @staticmethod
    def get_keypoints_from_paths(paths: List[Dict], 
                                 max_points: int = 1000) -> np.ndarray:
        """
        경로에서 특징점 추출
        
        Args:
            paths: 경로 정보 리스트
            max_points: 최대 특징점 수
            
        Returns:
            특징점 배열 (N, 2)
        """
        all_points = []
        
        for path in paths:
            coords = path['coords']
            # 경로의 시작점, 끝점, 중간점 등 추출
            if len(coords) > 0:
                all_points.extend(coords)
        
        if len(all_points) == 0:
            return np.array([]).reshape(0, 2)
        
        points = np.array(all_points)
        
        # 너무 많으면 샘플링
        if len(points) > max_points:
            indices = np.random.choice(len(points), max_points, replace=False)
            points = points[indices]
        
        return points


class VectorFeatureMatcher:
    """벡터 기반 특징점 매칭 클래스"""
    
    def __init__(self, use_transformer: bool = True):
        """
        Args:
            use_transformer: 트랜스포머 모델 사용 여부
        """
        self.use_transformer = use_transformer
        self.extractor = SVGPathExtractor()
        
        if use_transformer:
            # 경로 시퀀스를 임베딩하는 간단한 트랜스포머 사용
            # 실제로는 더 복잡한 모델을 사용할 수 있음
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def extract_features(self, svg_path: str) -> Dict:
        """
        SVG에서 벡터 특징 추출
        
        Args:
            svg_path: SVG 파일 경로
            
        Returns:
            특징 딕셔너리
        """
        paths = self.extractor.extract_paths_from_svg(svg_path)
        keypoints = self.extractor.get_keypoints_from_paths(paths)
        
        # 경로 통계 정보
        path_count = len(paths)
        total_points = sum(len(p['coords']) for p in paths)
        
        # 경계 박스
        if len(keypoints) > 0:
            bbox = {
                'min_x': keypoints[:, 0].min(),
                'max_x': keypoints[:, 0].max(),
                'min_y': keypoints[:, 1].min(),
                'max_y': keypoints[:, 1].max()
            }
        else:
            bbox = None
        
        return {
            'keypoints': keypoints,
            'paths': paths,
            'path_count': path_count,
            'total_points': total_points,
            'bbox': bbox
        }
    
    def match_vector_features(self, svg1_path: str, svg2_path: str) -> Dict:
        """
        두 SVG 파일 간 벡터 특징 매칭
        
        Args:
            svg1_path: 첫 번째 SVG 파일
            svg2_path: 두 번째 SVG 파일
            
        Returns:
            매칭 결과
        """
        feat1 = self.extract_features(svg1_path)
        feat2 = self.extract_features(svg2_path)
        
        kp1 = feat1['keypoints']
        kp2 = feat2['keypoints']
        
        if len(kp1) == 0 or len(kp2) == 0:
            return {
                'keypoints0': np.array([]),
                'keypoints1': np.array([]),
                'confidence': np.array([]),
                'num_matches': 0
            }
        
        # 특징점 정규화 (상대 좌표로 변환)
        # 각 SVG의 경계 박스로 정규화
        if len(kp1) > 0:
            kp1_min = kp1.min(axis=0)
            kp1_max = kp1.max(axis=0)
            kp1_range = kp1_max - kp1_min
            kp1_range[kp1_range == 0] = 1  # 0으로 나누기 방지
            kp1_norm = (kp1 - kp1_min) / kp1_range
        else:
            kp1_norm = kp1
        
        if len(kp2) > 0:
            kp2_min = kp2.min(axis=0)
            kp2_max = kp2.max(axis=0)
            kp2_range = kp2_max - kp2_min
            kp2_range[kp2_range == 0] = 1
            kp2_norm = (kp2 - kp2_min) / kp2_range
        else:
            kp2_norm = kp2
        
        # 거리 기반 매칭
        from scipy.spatial.distance import cdist
        
        if len(kp1_norm) == 0 or len(kp2_norm) == 0:
            return {
                'keypoints0': np.array([]),
                'keypoints1': np.array([]),
                'confidence': np.array([]),
                'num_matches': 0
            }
        
        distances = cdist(kp1_norm, kp2_norm)
        
        # 최근접 이웃 매칭
        matches = []
        match_distances = []
        
        # 상대적 거리 임계값 (정규화된 좌표에서)
        threshold = 0.1  # 정규화된 좌표에서 10% 이내
        
        for i in range(len(kp1_norm)):
            # 가장 가까운 점 찾기
            min_idx = np.argmin(distances[i])
            min_dist = distances[i, min_idx]
            
            # 거리 임계값 체크
            if min_dist < threshold:
                matches.append((i, min_idx))
                match_distances.append(min_dist)
        
        # 원본 좌표로 매칭점 반환
        if len(matches) > 0:
            pts1 = kp1[[m[0] for m in matches]]
            pts2 = kp2[[m[1] for m in matches]]
        else:
            pts1 = np.array([]).reshape(0, 2)
            pts2 = np.array([]).reshape(0, 2)
        
        if len(matches) < 4:
            return {
                'keypoints0': np.array([]),
                'keypoints1': np.array([]),
                'confidence': np.array([]),
                'num_matches': 0
            }
        
        # 매칭점 추출
        pts1 = kp1[[m[0] for m in matches]]
        pts2 = kp2[[m[1] for m in matches]]
        
        # 신뢰도 계산 (거리 기반)
        max_dist = max(match_distances) if match_distances else 1.0
        conf = np.array([1.0 - (d / max_dist) for d in match_distances])
        
        return {
            'keypoints0': pts1,
            'keypoints1': pts2,
            'confidence': conf,
            'num_matches': len(matches)
        }


class SVGTransformerMatcher:
    """트랜스포머 기반 SVG 경로 매칭"""
    
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.extractor = SVGPathExtractor()
    
    def path_to_sequence(self, paths: List[Dict]) -> List[int]:
        """
        경로를 시퀀스로 변환 (트랜스포머 입력용)
        
        Args:
            paths: 경로 정보 리스트
            
        Returns:
            토큰 시퀀스
        """
        # 간단한 버전: 경로의 좌표를 정규화하여 시퀀스로 변환
        sequence = []
        
        for path in paths:
            coords = path['coords']
            for x, y in coords[:50]:  # 최대 50개 좌표만 사용
                # 좌표를 정규화 (0-1000 범위로)
                x_norm = int((x / 4096) * 1000)
                y_norm = int((y / 3536) * 1000)
                sequence.extend([x_norm, y_norm])
        
        return sequence
    
    def match_with_transformer(self, svg1_path: str, svg2_path: str) -> Dict:
        """
        트랜스포머로 SVG 매칭 (향후 구현)
        
        현재는 벡터 기반 매칭 사용
        """
        matcher = VectorFeatureMatcher(use_transformer=False)
        return matcher.match_vector_features(svg1_path, svg2_path)

