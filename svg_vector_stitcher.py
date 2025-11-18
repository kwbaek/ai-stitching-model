"""
SVG 벡터 데이터를 직접 사용한 스티칭 모듈
"""
import numpy as np
import cv2
import xml.etree.ElementTree as ET
from typing import List, Tuple, Optional
from pathlib import Path

from svg_vector_analyzer import VectorFeatureMatcher, SVGPathExtractor
from image_aligner import ImageAligner


class SVGVectorStitcher:
    """SVG 벡터 데이터 직접 스티칭 클래스"""
    
    def __init__(self):
        self.matcher = VectorFeatureMatcher(use_transformer=False)
        self.aligner = ImageAligner()
        self.extractor = SVGPathExtractor()
    
    def apply_transform_to_svg(self, svg_path: str, transform_matrix: np.ndarray,
                              output_path: str, offset: Tuple[float, float] = (0, 0)):
        """
        SVG에 변환 행렬 적용
        
        Args:
            svg_path: 입력 SVG 파일
            transform_matrix: 3x3 변환 행렬
            output_path: 출력 SVG 파일
            offset: 오프셋 (x, y)
        """
        try:
            tree = ET.parse(svg_path)
            root = tree.getroot()
        except:
            return False
        
        # SVG 네임스페이스
        ns = {'svg': 'http://www.w3.org/2000/svg'}
        
        # 변환 행렬을 SVG transform 속성으로 변환
        # 간단한 버전: translate + matrix
        a, b, c, d, e, f = transform_matrix[0, 0], transform_matrix[0, 1], \
                          transform_matrix[1, 0], transform_matrix[1, 1], \
                          transform_matrix[0, 2] + offset[0], transform_matrix[1, 2] + offset[1]
        
        transform_str = f"matrix({a},{b},{c},{d},{e},{f})"
        
        # 모든 path에 transform 적용
        for path_elem in root.findall('.//path', ns):
            current_transform = path_elem.get('transform', '')
            if current_transform:
                new_transform = f"{transform_str} {current_transform}"
            else:
                new_transform = transform_str
            path_elem.set('transform', new_transform)
        
        # SVG 크기 조정
        width = root.get('width', '4096')
        height = root.get('height', '3536')
        
        # 새로운 크기 계산 (간단히)
        try:
            new_width = str(int(float(width)) * 2)  # 파노라마는 넓어짐
            root.set('width', new_width)
            root.set('viewBox', f"0 0 {new_width} {height}")
        except:
            pass
        
        # 저장
        tree.write(output_path, encoding='utf-8', xml_declaration=True)
        return True
    
    def stitch_svg_pair(self, svg1_path: str, svg2_path: str) -> Optional[np.ndarray]:
        """
        두 SVG 파일을 스티칭 (벡터 매칭 사용)
        
        Args:
            svg1_path: 첫 번째 SVG
            svg2_path: 두 번째 SVG
            
        Returns:
            호모그래피 행렬 또는 None
        """
        # 벡터 특징 매칭
        matches = self.matcher.match_vector_features(svg1_path, svg2_path)
        
        if matches['num_matches'] < 4:
            print(f"Not enough vector matches: {matches['num_matches']}")
            return None
        
        # 호모그래피 계산
        H = self.aligner.compute_homography(matches)
        
        return H
    
    def create_panorama_svg(self, svg_files: List[str], output_path: str,
                           max_images: Optional[int] = None) -> bool:
        """
        여러 SVG 파일을 파노라마 SVG로 합성
        
        Args:
            svg_files: SVG 파일 경로 리스트
            output_path: 출력 SVG 파일 경로
            max_images: 최대 이미지 수
        """
        if max_images:
            svg_files = svg_files[:max_images]
        
        if len(svg_files) < 2:
            return False
        
        # 첫 번째 SVG를 기준으로 시작
        base_svg = svg_files[0]
        
        try:
            tree = ET.parse(base_svg)
            root = tree.getroot()
        except:
            return False
        
        ns = {'svg': 'http://www.w3.org/2000/svg'}
        
        # 각 SVG를 순차적으로 추가
        current_offset_x = 0
        
        for i, svg_file in enumerate(svg_files[1:], 1):
            print(f"Stitching SVG {i+1}/{len(svg_files)}...")
            
            # 매칭 및 변환 행렬 계산
            H = self.stitch_svg_pair(base_svg, svg_file)
            
            if H is None:
                print(f"Warning: Failed to match SVG {i+1}, skipping...")
                continue
            
            # 두 번째 SVG의 경로 추출
            paths2 = self.extractor.extract_paths_from_svg(svg_file)
            
            # 변환 행렬 적용하여 경로 추가
            for path_info in paths2:
                coords = path_info['coords']
                
                # 좌표 변환
                if len(coords) > 0:
                    coords_array = np.array(coords, dtype=np.float32)
                    coords_homogeneous = np.column_stack([coords_array, np.ones(len(coords_array))])
                    transformed = (H @ coords_homogeneous.T).T
                    transformed_coords = transformed[:, :2] / transformed[:, 2:3]
                    
                    # 오프셋 적용
                    transformed_coords[:, 0] += current_offset_x
                    
                    # 새로운 path 요소 생성
                    path_d = 'M ' + ' L '.join([f"{x:.1f},{y:.1f}" for x, y in transformed_coords])
                    if path_info['d'].endswith('Z'):
                        path_d += ' Z'
                    
                    path_elem = ET.SubElement(root, 'path')
                    path_elem.set('d', path_d)
                    path_elem.set('fill', path_info.get('fill', 'lime'))
            
            # 오프셋 업데이트 (간단한 추정)
            current_offset_x += 3500  # 대략적인 이미지 너비
        
        # SVG 크기 조정
        try:
            width = root.get('width', '4096')
            new_width = str(int(float(width)) * len(svg_files))
            root.set('width', new_width)
            root.set('viewBox', f"0 0 {new_width} {root.get('height', '3536')}")
        except:
            pass
        
        # 저장
        tree.write(output_path, encoding='utf-8', xml_declaration=True)
        print(f"Panorama SVG saved to {output_path}")
        return True

