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
from svg_tile_layout import TileLayoutCalculator


class SVGVectorStitcher:
    """SVG 벡터 데이터 직접 스티칭 클래스"""
    
    def __init__(self):
        self.matcher = VectorFeatureMatcher(use_transformer=False)
        self.aligner = ImageAligner()
        self.extractor = SVGPathExtractor()
        self.tile_calculator = TileLayoutCalculator()
    
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
        여러 SVG 파일을 2D 타일 형태로 합성
        
        Args:
            svg_files: SVG 파일 경로 리스트
            output_path: 출력 SVG 파일 경로
            max_images: 최대 이미지 수
        """
        if max_images:
            svg_files = svg_files[:max_images]
        
        if len(svg_files) < 1:
            return False
        
        # 2D 타일 위치 계산
        print("Computing 2D tile layout...")
        positions = self.tile_calculator.compute_tile_positions(svg_files)
        positions = self.tile_calculator.normalize_positions(positions)
        
        # 그리드 크기 계산
        min_row, max_row, min_col, max_col = self.tile_calculator.calculate_grid_size(positions)
        num_rows = max_row - min_row + 1
        num_cols = max_col - min_col + 1
        
        print(f"Grid layout: {num_rows} rows x {num_cols} cols")
        
        # 빈 SVG 루트 생성
        try:
            # 첫 번째 SVG를 템플릿으로 사용
            template_svg = svg_files[0]
            tree = ET.parse(template_svg)
            root = tree.getroot()
            # 기존 모든 요소 제거 (path, circle 등)
            for elem in list(root):
                if elem.tag.endswith('path') or elem.tag.endswith('circle') or elem.tag.endswith('defs'):
                    root.remove(elem)
        except Exception as e:
            print(f"Error parsing template SVG: {e}")
            return False
        
        # SVG 크기 가져오기 (viewBox 또는 실제 bbox 사용)
        try:
            # viewBox에서 크기 가져오기
            viewbox = root.get('viewBox', '')
            if viewbox:
                parts = viewbox.split()
                if len(parts) >= 4:
                    img_width = int(float(parts[2]))
                    img_height = int(float(parts[3]))
                else:
                    img_width, img_height = 4096, 3536
            else:
                img_width = int(float(root.get('width', '4096')))
                img_height = int(float(root.get('height', '3536')))
        except:
            img_width, img_height = 4096, 3536
        
        print(f"Image size: {img_width} x {img_height}")
        print(f"Grid: {num_rows} rows x {num_cols} cols")
        
        # 각 이미지를 타일 위치에 배치
        for idx, svg_file in enumerate(svg_files):
            row, col = positions.get(idx, (idx // num_cols, idx % num_cols))
            print(f"Placing SVG {idx+1}/{len(svg_files)} at tile ({row}, {col})...")
            
            # 타일 위치에 맞는 오프셋 계산
            offset_x = col * img_width
            offset_y = row * img_height
            
            # SVG 파일 직접 파싱하여 path 요소 가져오기
            try:
                svg_tree = ET.parse(svg_file)
                svg_root = svg_tree.getroot()
            except Exception as e:
                print(f"Error parsing {svg_file}: {e}")
                continue
            
            import re
            
            # path 요소 찾기 (네임스페이스 처리)
            svg_paths = svg_root.findall('.//{http://www.w3.org/2000/svg}path') + svg_root.findall('.//path')
            
            for svg_path in svg_paths:
                path_d = svg_path.get('d', '')
                if not path_d:
                    continue
                
                # path d 문자열에서 좌표 추출 및 변환
                # 숫자 패턴 (음수 포함)
                def replace_coords(match):
                    x = float(match.group(1))
                    y = float(match.group(2))
                    # 오프셋 적용
                    x_new = x + offset_x
                    y_new = y + offset_y
                    return f"{x_new:.1f},{y_new:.1f}"
                
                # 좌표 쌍 찾아서 변환 (M, L, C, Q 등의 명령어 뒤의 좌표)
                # 패턴: 숫자,숫자 또는 숫자, 숫자
                path_d_transformed = re.sub(r'([-]?\d+\.?\d*),([-]?\d+\.?\d*)', replace_coords, path_d)
                
                # path 요소 생성
                path_elem = ET.SubElement(root, 'path')
                path_elem.set('d', path_d_transformed)
                path_elem.set('fill', svg_path.get('fill', 'lime'))
                
                # 다른 속성도 복사
                for attr in ['stroke', 'stroke-width', 'opacity']:
                    if svg_path.get(attr):
                        path_elem.set(attr, svg_path.get(attr))
            
            # circle 요소도 포함 (원본 SVG에 있는 경우)
            svg_circles = svg_root.findall('.//{http://www.w3.org/2000/svg}circle') + svg_root.findall('.//circle')
            
            for svg_circle in svg_circles:
                try:
                    cx = float(svg_circle.get('cx', '0'))
                    cy = float(svg_circle.get('cy', '0'))
                    r = svg_circle.get('r', '0')
                    fill = svg_circle.get('fill', 'red')
                    
                    # 오프셋 적용
                    cx_new = cx + offset_x
                    cy_new = cy + offset_y
                    
                    # circle 요소 생성
                    circle_elem = ET.SubElement(root, 'circle')
                    circle_elem.set('cx', f"{cx_new:.1f}")
                    circle_elem.set('cy', f"{cy_new:.1f}")
                    circle_elem.set('r', r)
                    circle_elem.set('fill', fill)
                    
                    # 다른 속성도 복사
                    for attr in ['stroke', 'stroke-width', 'opacity']:
                        if svg_circle.get(attr):
                            circle_elem.set(attr, svg_circle.get(attr))
                except (ValueError, TypeError):
                    continue
        
        # SVG 크기 조정 (타일 그리드에 맞게)
        try:
            canvas_width = num_cols * img_width
            canvas_height = num_rows * img_height
            root.set('width', str(canvas_width))
            root.set('height', str(canvas_height))
            root.set('viewBox', f"0 0 {canvas_width} {canvas_height}")
        except Exception as e:
            print(f"Error setting SVG size: {e}")
        
        # 저장
        tree.write(output_path, encoding='utf-8', xml_declaration=True)
        print(f"Tiled panorama SVG saved to {output_path}")
        print(f"Canvas size: {canvas_width} x {canvas_height}")
        return True

