"""
SVG 벡터 데이터를 직접 사용한 스티칭 모듈
"""
import numpy as np
import cv2
import xml.etree.ElementTree as ET
from typing import List, Tuple, Optional, Dict
from pathlib import Path

from svg_vector_analyzer import VectorFeatureMatcher, SVGPathExtractor
from image_aligner import ImageAligner
from svg_tile_layout import TileLayoutCalculator
from svg_overlap_detector import OverlapDetector
try:
    from transformer_vector_matcher import TransformerVectorMatcher
    TRANSFORMER_AVAILABLE = True
except ImportError:
    TRANSFORMER_AVAILABLE = False
    print("Warning: TransformerVectorMatcher not available")


class SVGVectorStitcher:
    """SVG 벡터 데이터 직접 스티칭 클래스"""
    
    def __init__(self, use_transformer: bool = False, 
                 use_overlap_detection: bool = True,
                 layout_mode: str = 'auto'):
        """
        Args:
            use_transformer: 트랜스포머 기반 매칭 사용 여부
            use_overlap_detection: 겹침 감지 사용 여부
            layout_mode: 레이아웃 모드 ('auto', 'horizontal', 'vertical', 'grid')
        """
        self.use_transformer = use_transformer and TRANSFORMER_AVAILABLE
        self.use_overlap_detection = use_overlap_detection
        self.layout_mode = layout_mode
        
        # 기본 매처
        self.matcher = VectorFeatureMatcher(use_transformer=False)
        self.aligner = ImageAligner()
        self.extractor = SVGPathExtractor()
        self.tile_calculator = TileLayoutCalculator(self.matcher, self.aligner)
        
        # 겹침 감지기
        if use_overlap_detection:
            self.overlap_detector = OverlapDetector(overlap_threshold=0.1)
        
        # 트랜스포머 매처
        if self.use_transformer:
            self.transformer_matcher = TransformerVectorMatcher()
            print("Using transformer-based vector matching")
        else:
            self.transformer_matcher = None
    
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
        # 트랜스포머 매칭 사용
        if self.use_transformer and self.transformer_matcher:
            matches = self.transformer_matcher.match_vectors(svg1_path, svg2_path)
        else:
            # 기본 벡터 특징 매칭
            matches = self.matcher.match_vector_features(svg1_path, svg2_path)
        
        if matches['num_matches'] < 4:
            print(f"Not enough vector matches: {matches['num_matches']}")
            return None
        
        # 호모그래피 계산
        H = self.aligner.compute_homography(matches)
        
        return H
    
    def determine_layout_mode(self, svg_files: List[str]) -> str:
        """
        SVG 파일들의 겹침 패턴을 분석하여 최적 레이아웃 모드 결정
        
        Args:
            svg_files: SVG 파일 경로 리스트
            
        Returns:
            레이아웃 모드 ('horizontal', 'vertical', 'grid')
        """
        if not self.use_overlap_detection or len(svg_files) < 2:
            return 'grid'
        
        # 연속적인 겹침 분석
        overlaps = self.overlap_detector.analyze_sequence_overlaps(svg_files[:min(10, len(svg_files))])
        
        # 방향 통계
        directions = [o['direction'] for o in overlaps if o['has_overlap']]
        
        if not directions:
            return 'grid'
        
        # 수평/수직 방향 카운트
        horizontal_count = sum(1 for d in directions if d in ['left', 'right'])
        vertical_count = sum(1 for d in directions if d in ['top', 'bottom'])
        
        # 주요 방향 결정
        if horizontal_count > vertical_count * 1.5:
            return 'horizontal'
        elif vertical_count > horizontal_count * 1.5:
            return 'vertical'
        else:
            return 'grid'
    
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
        
        # 연속적인 이미지들 간의 호모그래피를 계산하여 누적 변환 적용
        print("Computing sequential homography-based panorama alignment...")
        
        # 인접한 이미지들 간의 호모그래피 계산
        pairwise_homographies = {}
        for i in range(len(svg_files) - 1):
            matches = self.matcher.match_vector_features(svg_files[i], svg_files[i + 1])
            if matches['num_matches'] >= 4:
                H = self.aligner.compute_homography(matches)
                if H is not None:
                    pairwise_homographies[i] = H  # i -> i+1 변환
                    print(f"  Image {i+1} -> {i+2}: {matches['num_matches']} matches, homography computed")
                else:
                    print(f"  Image {i+1} -> {i+2}: Failed to compute homography")
            else:
                print(f"  Image {i+1} -> {i+2}: Only {matches['num_matches']} matches (need >= 4)")
        
        # 누적 변환 행렬 계산 (첫 번째 이미지 기준)
        cumulative_homographies = {0: np.eye(3)}  # 첫 번째 이미지는 단위 행렬
        
        # 첫 번째 이미지의 bbox로 이미지 크기 추정
        feat0 = self.matcher.extract_features(svg_files[0])
        if feat0['bbox'] is None:
            img_width, img_height = 4096, 3536
        else:
            img_width = int(feat0['bbox']['max_x'] - feat0['bbox']['min_x'])
            img_height = int(feat0['bbox']['max_y'] - feat0['bbox']['min_y'])
        
        # 각 이미지의 위치 오프셋 계산 (호모그래피로 상대 위치 계산, 하지만 이미지 자체는 변형 없음)
        image_offsets = {0: (0, 0)}  # 첫 번째 이미지는 (0, 0)
        
        for i in range(1, len(svg_files)):
            if (i - 1) in pairwise_homographies:
                # 이전 이미지에서 현재 이미지로의 호모그래피
                H_prev_to_curr = pairwise_homographies[i - 1]
                
                # 이전 이미지의 중심점을 현재 이미지 좌표계로 변환
                prev_center = np.array([[img_width/2, img_height/2, 1]], dtype=np.float32).T
                prev_center_in_curr = H_prev_to_curr @ prev_center
                prev_center_in_curr = prev_center_in_curr / prev_center_in_curr[2, 0]
                
                # 현재 이미지의 중심점
                curr_center = np.array([img_width/2, img_height/2])
                
                # 상대적 위치 차이 계산
                relative_offset_x = prev_center_in_curr[0, 0] - curr_center[0]
                relative_offset_y = prev_center_in_curr[1, 0] - curr_center[1]
                
                # 이전 이미지의 오른쪽 가장자리를 현재 이미지 좌표계로 변환하여 겹침 확인
                prev_right_edge = np.array([[img_width, img_height/2, 1]], dtype=np.float32).T
                prev_right_in_curr = H_prev_to_curr @ prev_right_edge
                prev_right_in_curr = prev_right_in_curr / prev_right_in_curr[2, 0]
                
                # 현재 이미지의 왼쪽 가장자리 (x=0)
                overlap_amount = prev_right_in_curr[0, 0] - 0
                
                # 목표: 약 10% 겹침 (img_width * 0.1)
                target_overlap = img_width * 0.1
                
                # 누적 오프셋 계산 (이전 이미지의 오프셋 + 상대적 위치 차이)
                prev_offset_x, prev_offset_y = image_offsets[i - 1]
                
                # 겹침을 고려한 오프셋 조정
                if overlap_amount > img_width * 0.2:
                    # 겹침이 너무 크면 오른쪽으로 이동
                    offset_x = prev_offset_x + img_width - target_overlap
                elif overlap_amount < 0:
                    # 겹침이 없으면 약간 겹치도록
                    offset_x = prev_offset_x + img_width - target_overlap
                else:
                    # 적절한 겹침이면 상대적 위치 사용
                    offset_x = prev_offset_x + relative_offset_x + (img_width - overlap_amount)
                
                # Y축은 상대적 위치 사용 (수직 정렬)
                offset_y = prev_offset_y + relative_offset_y
                
                image_offsets[i] = (offset_x, offset_y)
                print(f"  Image {i+1}: Offset ({offset_x:.1f}, {offset_y:.1f}), overlap: {overlap_amount/img_width*100:.1f}%")
            else:
                # 매칭 실패 시, 이전 이미지 옆에 배치
                prev_offset_x, prev_offset_y = image_offsets[i - 1]
                image_offsets[i] = (prev_offset_x + img_width * 0.9, prev_offset_y)  # 10% 겹침
                print(f"  Image {i+1}: No match, placed next to previous")
        
        # img_width, img_height는 이미 위에서 계산됨
        
        # 모든 이미지의 위치를 계산하여 캔버스 크기 결정
        all_corners_x = [0, img_width]
        all_corners_y = [0, img_height]
        
        for i, (offset_x, offset_y) in image_offsets.items():
            # 이미지의 네 모서리 (오프셋만 적용, 변형 없음)
            all_corners_x.extend([offset_x, offset_x + img_width])
            all_corners_y.extend([offset_y, offset_y + img_height])
        
        # 캔버스 크기 계산
        min_x = min(all_corners_x)
        max_x = max(all_corners_x)
        min_y = min(all_corners_y)
        max_y = max(all_corners_y)
        
        # 모든 좌표가 양수가 되도록 오프셋 계산
        offset_x = -min_x if min_x < 0 else 0
        offset_y = -min_y if min_y < 0 else 0
        
        # 오프셋 적용 후 다시 계산
        adjusted_max_x = max_x + offset_x
        adjusted_max_y = max_y + offset_y
        
        canvas_width = int(adjusted_max_x) + img_width
        canvas_height = int(adjusted_max_y) + img_height
        
        print(f"Canvas size: {canvas_width} x {canvas_height}")
        print(f"Offset: ({offset_x}, {offset_y})")
        
        # 빈 SVG 루트 생성
        try:
            # 첫 번째 SVG를 템플릿으로 사용
            template_svg = svg_files[0]
            tree = ET.parse(template_svg)
            root = tree.getroot()
            # 기존 모든 요소 제거 (path, circle 등)
            for elem in list(root):
                if elem.tag.endswith('path') or elem.tag.endswith('circle') or elem.tag.endswith('defs') or elem.tag.endswith('rect') or elem.tag.endswith('text'):
                    root.remove(elem)
            
            # SVG 네임스페이스 명시적으로 설정 (중복 방지 - 기존 속성 제거 후 재설정)
            # 기존 xmlns 속성 제거
            if 'xmlns' in root.attrib:
                del root.attrib['xmlns']
            if 'xmlns:xlink' in root.attrib:
                del root.attrib['xmlns:xlink']
            
            # 새로 설정
            root.set('xmlns', 'http://www.w3.org/2000/svg')
            root.set('xmlns:xlink', 'http://www.w3.org/1999/xlink')
        except Exception as e:
            print(f"Error parsing template SVG: {e}")
            # 템플릿 파싱 실패 시 새로 생성
            root = ET.Element('svg')
            root.set('xmlns', 'http://www.w3.org/2000/svg')
            root.set('xmlns:xlink', 'http://www.w3.org/1999/xlink')
            tree = ET.ElementTree(root)
        
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
        
        # 각 이미지를 원본 그대로 배치 (변형 없이 오프셋만 적용)
        for idx, svg_file in enumerate(svg_files):
            print(f"Placing SVG {idx+1}/{len(svg_files)}...")
            
            # 이미지 오프셋 가져오기
            if idx in image_offsets:
                img_offset_x, img_offset_y = image_offsets[idx]
                base_offset_x = offset_x + img_offset_x
                base_offset_y = offset_y + img_offset_y
            else:
                # 오프셋이 없으면 기본 위치
                base_offset_x = offset_x
                base_offset_y = offset_y
            
            # 파일명 표시 (디버깅용, 작게)
            from pathlib import Path
            file_name = Path(svg_file).stem
            text_elem = ET.SubElement(root, 'text')
            text_elem.set('x', str(base_offset_x + 20))
            text_elem.set('y', str(base_offset_y + 50))
            text_elem.set('fill', 'blue')
            text_elem.set('font-size', '40')
            text_elem.set('font-weight', 'bold')
            text_elem.set('font-family', 'Arial, sans-serif')
            text_elem.set('opacity', '0.5')
            text_elem.text = f"{file_name}"
            
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
                
                # path d 문자열에서 좌표 추출 및 변환 (이미지 변형 없이 오프셋만 적용)
                def replace_coords(match):
                    x = float(match.group(1))
                    y = float(match.group(2))
                    
                    # 단순 오프셋만 적용 (이미지 자체는 변형 없음)
                    x_new = x + base_offset_x
                    y_new = y + base_offset_y
                    
                    return f"{x_new:.1f},{y_new:.1f}"
                
                # 좌표 쌍 찾아서 변환
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
                    
                    # 단순 오프셋만 적용 (이미지 자체는 변형 없음)
                    cx_new = cx + base_offset_x
                    cy_new = cy + base_offset_y
                    
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
        
        # SVG 크기 조정 (계산된 캔버스 크기에 맞게)
        try:
            root.set('width', str(canvas_width))
            root.set('height', str(canvas_height))
            root.set('viewBox', f"0 0 {canvas_width} {canvas_height}")
        except Exception as e:
            print(f"Error setting SVG size: {e}")
        
        # 저장 (XML 선언 및 네임스페이스 명시)
        # ET.register_namespace는 ET.write()가 자동으로 xmlns를 추가하므로 사용하지 않음
        
        # XML 선언과 함께 저장
        tree.write(output_path, encoding='utf-8', xml_declaration=True, method='xml')
        
        # 저장 후 네임스페이스 형식 정리 (xmlns:ns0 -> xmlns)
        with open(output_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        import re
        # xmlns:ns0 형식을 일반 xmlns로 변경
        content = re.sub(r'xmlns:ns0="http://www\.w3\.org/2000/svg"', 'xmlns="http://www.w3.org/2000/svg"', content)
        # <ns0:svg> -> <svg>, </ns0:svg> -> </svg>
        content = re.sub(r'<ns0:svg', '<svg', content)
        content = re.sub(r'</ns0:svg>', '</svg>', content)
        # 중복된 xmlns 속성 제거
        svg_tag_match = re.search(r'<svg\s+([^>]*)>', content)
        if svg_tag_match:
            attrs = svg_tag_match.group(1)
            # xmlns가 여러 번 나타나는지 확인
            xmlns_matches = list(re.finditer(r'xmlns="http://www\.w3\.org/2000/svg"', attrs))
            if len(xmlns_matches) > 1:
                # 첫 번째 xmlns만 유지하고 나머지 제거
                for match in reversed(xmlns_matches[1:]):
                    start, end = match.span()
                    # 앞뒤 공백도 함께 제거
                    if start > 0 and attrs[start-1] == ' ':
                        start -= 1
                    attrs = attrs[:start] + attrs[end:]
                content = re.sub(r'<svg\s+[^>]*>', f'<svg {attrs}>', content, count=1)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"Panorama SVG saved to {output_path}")
        print(f"Canvas size: {canvas_width} x {canvas_height}")
        return True

