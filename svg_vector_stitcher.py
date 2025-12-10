"""
SVG 벡터 데이터를 직접 사용한 스티칭 모듈
"""
import numpy as np
import cv2
import xml.etree.ElementTree as ET
from typing import List, Tuple, Optional, Dict
from pathlib import Path
import os
import multiprocessing as mp

from svg_vector_analyzer import VectorFeatureMatcher, SVGPathExtractor
from image_aligner import ImageAligner
from svg_tile_layout import TileLayoutCalculator
from svg_overlap_detector import OverlapDetector
from svg_converter import SVGConverter
from feature_matcher import DeepFeatureMatcher, TraditionalFeatureMatcher
from vector_refinement import VectorRefiner
try:
    from transformer_vector_matcher import TransformerVectorMatcher
    TRANSFORMER_AVAILABLE = True
except ImportError:
    TRANSFORMER_AVAILABLE = False
    print("Warning: TransformerVectorMatcher not available")

try:
    from graph_vector_matcher import GraphVectorMatcher
    GNN_AVAILABLE = True
except ImportError:
    GNN_AVAILABLE = False
    print("Warning: GraphVectorMatcher not available (torch-geometric required)")


def _match_pairs_worker(args):
    """
    Worker function for parallel matching on a specific GPU
    
    Args:
        args: tuple of (gpu_id, pairs, svg_files, raster_method, converter_params, refiner_params, full_size_params)
    
    Returns:
        Dictionary of matching results for assigned pairs
    """
    import sys
    gpu_id, pairs, svg_files, raster_method, converter_params, refiner_params, full_size_params = args
    
    # Use direct CUDA device (cuda:0 or cuda:1)
    device = f'cuda:{gpu_id}'
    
    print(f"Worker GPU {gpu_id} starting with {len(pairs)} pairs on {device}...", flush=True)
    sys.stdout.flush()
    
    # Initialize components in worker process
    try:
        import torch
        from svg_converter import SVGConverter
        from feature_matcher import DeepFeatureMatcher
        from vector_refinement import VectorRefiner
        from image_aligner import ImageAligner
        
        print(f"  GPU {gpu_id}: Imports successful", flush=True)
        sys.stdout.flush()
        
        converter = SVGConverter(**converter_params)
        full_size_converter = SVGConverter(**full_size_params)
        refiner = VectorRefiner(**refiner_params)
        aligner = ImageAligner()
        
        print(f"  GPU {gpu_id}: Initializing LoFTR model on {device}...", flush=True)
        sys.stdout.flush()
        
        raster_matcher = DeepFeatureMatcher(method=raster_method, device=device)
        
        print(f"  GPU {gpu_id}: Model loaded successfully", flush=True)
        sys.stdout.flush()
    except Exception as e:
        print(f"  GPU {gpu_id}: Initialization error: {e}", flush=True)
        sys.stdout.flush()
        return {}
    
    results = {}
    
    # Progress tracking
    progress_file = f'/app/data/ai-stitching-model/progress_gpu_{gpu_id}.json'
    import json
    
    last_match_count = 0
    last_matrix = "Identity"
    
    for idx, (i, j) in enumerate(pairs):
        # Update progress every pair
        progress_data = {
            'gpu_id': gpu_id,
            'current_pair': idx + 1,
            'total_pairs': len(pairs),
            'progress_percent': ((idx + 1) / len(pairs)) * 100,
            'current_images': [i+1, j+1],
            'completed_matches': len(results),
            'current_svg_files': [
                os.path.relpath(svg_files[i], '/app/data/ai-stitching-model'),
                os.path.relpath(svg_files[j], '/app/data/ai-stitching-model')
            ],
            'last_metrics': {
                'matches': last_match_count,
                'matrix': last_matrix
            }
        }
        
        try:
            with open(progress_file, 'w') as f:
                json.dump(progress_data, f)
        except:
            pass
        
        if idx % 10 == 0:
            print(f"  GPU {gpu_id}: Processing pair {idx}/{len(pairs)} ({i+1}<->{j+1})", flush=True)
            sys.stdout.flush()
        
        try:
            # Convert SVG to raster
            img1 = converter.svg_to_image(svg_files[i])
            img2 = converter.svg_to_image(svg_files[j])
            
            # Match features
            matches = raster_matcher.match_features(img1, img2)
            
            # Update metrics for next iteration
            last_match_count = matches['num_matches']
            
            # Free memory
            del img1, img2
            torch.cuda.empty_cache()
            
            # Compute homography if enough matches
            if matches['num_matches'] >= 10:
                H_low = aligner.compute_homography(matches)
                if H_low is not None:
                    # Scale homography
                    scale_x = full_size_converter.output_size[0] / converter.output_size[0]
                    scale_y = full_size_converter.output_size[1] / converter.output_size[1]
                    S = np.diag([scale_x, scale_y, 1.0])
                    S_inv = np.diag([1.0/scale_x, 1.0/scale_y, 1.0])
                    H = S @ H_low @ S_inv
                    
                    # Refine
                    try:
                        H = refiner.refine_alignment(svg_files[i], svg_files[j], H, max_distance=50.0, translation_only=True)
                    except:
                        pass
                    
                    results[(i, j)] = {
                        'H': H,
                        'matches': matches['num_matches']
                    }
                    
                    # Format matrix for display
                    h_flat = H.flatten()
                    last_matrix = f"[{h_flat[0]:.2f}, {h_flat[1]:.2f}, {h_flat[2]:.2f}, ...]"
                else:
                    last_matrix = "Failed to compute H"
            else:
                last_matrix = "Insufficient matches"
                
        except Exception as e:
            print(f"  GPU {gpu_id}: Error matching {i+1}<->{j+1}: {e}", flush=True)
            sys.stdout.flush()
            continue
    
    print(f"Worker GPU {gpu_id} completed {len(results)} successful matches", flush=True)
    sys.stdout.flush()
    return results



class SVGVectorStitcher:
    """SVG 벡터 데이터 직접 스티칭 클래스"""
    
    def __init__(self, use_transformer: bool = False,
                 use_gnn: bool = False,
                 use_overlap_detection: bool = True,
                 layout_mode: str = 'auto',
                 use_raster_matching: bool = True,
                 raster_method: str = 'loftr',
                 show_labels: bool = False,
                 show_borders: bool = False):
        """
        Args:
            use_transformer: 트랜스포머 기반 벡터 매칭 사용 여부
            use_gnn: Graph Neural Network 기반 벡터 매칭 사용 여부
            use_overlap_detection: 겹침 감지 사용 여부
            layout_mode: 레이아웃 모드 ('auto', 'horizontal', 'vertical', 'grid')
            use_raster_matching: 래스터 기반 딥러닝 매칭 사용 (권장) ⭐
            raster_method: 래스터 매칭 방법 ('loftr', 'disk', 'lightglue', 'lightglue_disk', 'dinov2')
            show_labels: 파일명 라벨 표시 여부
            show_borders: 타일 경계선 표시 여부
        """
        self.use_transformer = use_transformer and TRANSFORMER_AVAILABLE
        self.use_gnn = use_gnn and GNN_AVAILABLE
        self.use_overlap_detection = use_overlap_detection
        self.layout_mode = layout_mode
        self.use_raster_matching = use_raster_matching
        self.raster_method = raster_method
        self.show_labels = show_labels
        self.show_borders = show_borders
        
        # SVG → 래스터 변환기 (메모리 절약을 위해 크기 조정)
        # 원본 크기 유지하되, 딥러닝 매칭용으로는 작은 크기 사용
        # LoFTR 등 트랜스포머 모델은 메모리 사용량이 크므로 해상도를 낮춤 (1024px 수준)
        self.converter = SVGConverter(output_size=(1024, 884))  # 1/4 크기 (메모리 최적화)
        self.refiner = VectorRefiner()
        self.full_size_converter = SVGConverter(output_size=(4096, 3536))  # 최종 출력용
        
        # 래스터 기반 딥러닝 매처 (권장) ⭐
        if use_raster_matching:
            try:
                self.raster_matcher = DeepFeatureMatcher(method=raster_method)
                print(f"Using raster-based deep learning matching: {raster_method}")
            except Exception as e:
                print(f"Warning: Failed to initialize {raster_method}, falling back to vector matching")
                self.raster_matcher = None
                use_raster_matching = False
        else:
            self.raster_matcher = None
        
        # 벡터 기반 매처 (fallback)
        self.matcher = VectorFeatureMatcher(use_transformer=False)
        self.aligner = ImageAligner()
        self.extractor = SVGPathExtractor()
        self.tile_calculator = TileLayoutCalculator(self.matcher, self.aligner)
        
        # 겹침 감지기
        if use_overlap_detection:
            self.overlap_detector = OverlapDetector(overlap_threshold=0.1)
        
        # 트랜스포머 매처 (벡터용)
        if self.use_transformer:
            self.transformer_matcher = TransformerVectorMatcher()
            print("Using transformer-based vector matching (fallback)")
        else:
            self.transformer_matcher = None
        
        # GNN 매처 (벡터용)
        if self.use_gnn:
            self.gnn_matcher = GraphVectorMatcher()
            print("Using Graph Neural Network-based vector matching (fallback)")
        else:
            self.gnn_matcher = None
    
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
        # 매칭 방법 선택 (우선순위: GNN > Transformer > 기본)
        if self.use_gnn and self.gnn_matcher:
            matches = self.gnn_matcher.match_graphs(svg1_path, svg2_path)
        elif self.use_transformer and self.transformer_matcher:
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
        
        # 모든 이미지 쌍 간의 호모그래피 계산 (상하좌우 모든 방향)
        print("Computing 2D grid layout with all-directional relationships...")
        
        # 모든 이미지 쌍 간의 호모그래피 계산
        all_homographies = {}
        relationships = {}  # {(i, j): {'direction': 'right'|'left'|'up'|'down', 'offset_x': float, 'offset_y': float}}
        
        n = len(svg_files)
        print(f"Analyzing relationships between {n} images...")
        
        # Use multiprocessing to parallelize matching across 2 GPUs
        if self.use_raster_matching and self.raster_matcher:
            # Generate all pairs
            all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
            total_pairs = len(all_pairs)
            print(f"Total pairs to match: {total_pairs}")
            
            # Split pairs across 2 GPUs
            pairs_per_gpu = total_pairs // 2
            gpu_0_pairs = all_pairs[:pairs_per_gpu]
            gpu_1_pairs = all_pairs[pairs_per_gpu:]
            
            # Prepare parameters for workers
            converter_params = {'output_size': self.converter.output_size}
            full_size_params = {'output_size': self.full_size_converter.output_size}
            refiner_params = {'max_iterations': self.refiner.max_iterations, 'tolerance': self.refiner.tolerance}
            
            worker_args = [
                (0, gpu_0_pairs, svg_files, self.raster_method, converter_params, refiner_params, full_size_params),
                (1, gpu_1_pairs, svg_files, self.raster_method, converter_params, refiner_params, full_size_params)
            ]
            
            print(f"GPU 0: {len(gpu_0_pairs)} pairs, GPU 1: {len(gpu_1_pairs)} pairs")
            print("Starting parallel matching on 2 GPUs...")
            
            # Set spawn method for CUDA compatibility
            try:
                mp.set_start_method('spawn', force=True)
            except RuntimeError:
                pass  # Already set
            
            # Run matching in parallel
            with mp.Pool(processes=2) as pool:
                worker_results = pool.map(_match_pairs_worker, worker_args)
            
            # Merge results from both workers
            for worker_result in worker_results:
                for (i, j), match_data in worker_result.items():
                    H = match_data['H']
                    num_matches = match_data['matches']
                    
                    all_homographies[(i, j)] = H
                    try:
                        all_homographies[(j, i)] = np.linalg.inv(H)
                    except np.linalg.LinAlgError:
                        pass
                    
                    relationships[(i, j)] = {'homography': H, 'matches': num_matches}
                    relationships[(j, i)] = {'homography': all_homographies.get((j, i)), 'matches': num_matches}
                    
                    print(f"  Image {i+1} <-> {j+1}: {num_matches} matches (loftr)")
            
            print(f"Parallel matching complete. Total matches: {len(all_homographies) // 2}")
        else:
            # Fallback to sequential vector matching
            for i in range(n):
                for j in range(i + 1, n):
                    matches = self.matcher.match_vector_features(svg_files[i], svg_files[j])
                    if matches['num_matches'] >= 4:
                        H = self.aligner.compute_homography(matches)
                        if H is not None:
                            all_homographies[(i, j)] = H
                            try:
                                all_homographies[(j, i)] = np.linalg.inv(H)
                            except np.linalg.LinAlgError:
                                pass
                            
                            relationships[(i, j)] = {'homography': H, 'matches': matches['num_matches']}
                            relationships[(j, i)] = {'homography': all_homographies.get((j, i)), 'matches': matches['num_matches']}
                            
                            if matches['num_matches'] >= 10:
                                print(f"  Image {i+1} <-> {j+1}: {matches['num_matches']} matches (vector)")

        
        # 첫 번째 이미지의 bbox로 이미지 크기 추정
        feat0 = self.matcher.extract_features(svg_files[0])
        if feat0['bbox'] is None:
            img_width, img_height = 4096, 3536
        else:
            img_width = int(feat0['bbox']['max_x'] - feat0['bbox']['min_x'])
            img_height = int(feat0['bbox']['max_y'] - feat0['bbox']['min_y'])
        
        # 상대 위치 분석 (이미지 크기 정보 필요)
        analyzed_relationships = {}
        for (i, j), rel_data in relationships.items():
            if rel_data.get('homography') is not None:
                rel_info = self._analyze_relative_position(svg_files[i], svg_files[j], rel_data['homography'], img_width, img_height)
                analyzed_relationships[(i, j)] = rel_info
        
        # 2D 그리드 위치 계산
        print("Computing 2D grid positions...")
        grid_positions = self._compute_grid_positions(n, analyzed_relationships)
        
        print(f"Grid layout: {len(set(grid_positions.values()))} unique positions")
        
        # 그리드 기반 배치: 각 이미지의 위치 계산
        image_offsets = {}  # {idx: (x, y)}
        image_transforms = {}  # {idx: transform_matrix}
        
        # 목표: 약 10% 겹침
        target_overlap_ratio = 0.1
        target_overlap = img_width * target_overlap_ratio
        
        # 순차적 스티칭: 첫 번째 이미지를 기준으로 가장 가까운 이미지부터 순차적으로 배치
        # 첫 번째 이미지를 (0, 0)에 배치
        image_offsets[0] = (0, 0)
        image_transforms[0] = np.eye(3)
        
        # 이미지 간 거리 계산 (매칭 점 수 기반)
        image_distances = {}
        for (i, j), rel_data in relationships.items():
            if rel_data.get('homography') is not None:
                matches_count = rel_data.get('matches', 0)
                # 거리 = 1 / (매칭 점 수 + 1) - 매칭이 많을수록 가까움
                distance = 1.0 / (matches_count + 1)
                image_distances[(i, j)] = distance
                image_distances[(j, i)] = distance
        
        # BFS로 그리드를 순회하며 위치 계산 (가장 가까운 이미지부터)
        visited = {0}
        queue = [0]
        
        while queue:
            current_idx = queue.pop(0)
            current_row, current_col = grid_positions[current_idx]
            current_offset_x, current_offset_y = image_offsets[current_idx]
            current_transform = image_transforms[current_idx]
            
            # 상하좌우 이웃 찾기
            neighbors = [
                (1, 0, 'right'),   # 오른쪽
                (-1, 0, 'left'),   # 왼쪽
                (0, 1, 'down'),    # 아래
                (0, -1, 'up')      # 위
            ]
            
            for dr, dc, direction in neighbors:
                neighbor_row = current_row + dr
                neighbor_col = current_col + dc
                
                # 이 위치에 있는 이미지 찾기
                neighbor_idx = None
                for idx, (row, col) in grid_positions.items():
                    if row == neighbor_row and col == neighbor_col and idx not in visited:
                        neighbor_idx = idx
                        break
                
                if neighbor_idx is None:
                    continue
                
                # 호모그래피 가져오기
                if (current_idx, neighbor_idx) in all_homographies:
                    H = all_homographies[(current_idx, neighbor_idx)]
                elif (neighbor_idx, current_idx) in all_homographies:
                    try:
                        H = np.linalg.inv(all_homographies[(neighbor_idx, current_idx)])
                    except np.linalg.LinAlgError:
                        continue
                else:
                    continue
                
                # 방향에 따라 오프셋 계산
                if direction == 'right':
                    # 현재 이미지의 오른쪽 가장자리
                    prev_right = current_transform @ np.array([[img_width], [img_height/2], [1]], dtype=np.float32)
                    prev_right = prev_right / prev_right[2, 0]
                    prev_right_x = prev_right[0, 0]
                    prev_right_y = prev_right[1, 0]
                    
                    # 이웃 이미지의 왼쪽 가장자리를 현재 이미지 좌표계로 변환
                    try:
                        H_neighbor_to_current = np.linalg.inv(H)
                    except np.linalg.LinAlgError:
                        H_neighbor_to_current = np.eye(3)
                    
                    neighbor_left = H_neighbor_to_current @ np.array([[0], [img_height/2], [1]], dtype=np.float32)
                    neighbor_left = neighbor_left / neighbor_left[2, 0]
                    
                    # 목표: 이웃 이미지의 왼쪽이 현재 이미지의 오른쪽에서 target_overlap만큼 왼쪽에 위치
                    target_left_x = prev_right_x - target_overlap
                    
                    # Y축은 호모그래피로 계산한 상대적 위치 사용
                    prev_center = current_transform @ np.array([[img_width/2], [img_height/2], [1]], dtype=np.float32)
                    prev_center = prev_center / prev_center[2, 0]
                    
                    neighbor_center_in_current = H_neighbor_to_current @ np.array([[img_width/2], [img_height/2], [1]], dtype=np.float32)
                    neighbor_center_in_current = neighbor_center_in_current / neighbor_center_in_current[2, 0]
                    
                    relative_offset_y = neighbor_center_in_current[1, 0] - img_height/2
                    target_center_y = prev_center[1, 0] + relative_offset_y
                    
                    neighbor_offset_x = target_left_x
                    neighbor_offset_y = target_center_y - img_height / 2
                    
                elif direction == 'left':
                    # 현재 이미지의 왼쪽 가장자리
                    prev_left = current_transform @ np.array([[0], [img_height/2], [1]], dtype=np.float32)
                    prev_left = prev_left / prev_left[2, 0]
                    prev_left_x = prev_left[0, 0]
                    
                    # 이웃 이미지의 오른쪽 가장자리를 현재 이미지 좌표계로 변환
                    neighbor_right = H @ np.array([[img_width], [img_height/2], [1]], dtype=np.float32)
                    neighbor_right = neighbor_right / neighbor_right[2, 0]
                    
                    # 목표: 이웃 이미지의 오른쪽이 현재 이미지의 왼쪽에서 target_overlap만큼 오른쪽에 위치
                    target_right_x = prev_left_x + target_overlap
                    
                    # Y축은 호모그래피로 계산한 상대적 위치 사용
                    prev_center = current_transform @ np.array([[img_width/2], [img_height/2], [1]], dtype=np.float32)
                    prev_center = prev_center / prev_center[2, 0]
                    
                    neighbor_center_in_current = H @ np.array([[img_width/2], [img_height/2], [1]], dtype=np.float32)
                    neighbor_center_in_current = neighbor_center_in_current / neighbor_center_in_current[2, 0]
                    
                    relative_offset_y = neighbor_center_in_current[1, 0] - img_height/2
                    target_center_y = prev_center[1, 0] + relative_offset_y
                    
                    neighbor_offset_x = target_right_x - img_width
                    neighbor_offset_y = target_center_y - img_height / 2
                    
                elif direction == 'down':
                    # 현재 이미지의 아래 가장자리
                    prev_bottom = current_transform @ np.array([[img_width/2], [img_height], [1]], dtype=np.float32)
                    prev_bottom = prev_bottom / prev_bottom[2, 0]
                    prev_bottom_y = prev_bottom[1, 0]
                    
                    # 이웃 이미지의 위 가장자리를 현재 이미지 좌표계로 변환
                    try:
                        H_neighbor_to_current = np.linalg.inv(H)
                    except np.linalg.LinAlgError:
                        H_neighbor_to_current = np.eye(3)
                    
                    neighbor_top = H_neighbor_to_current @ np.array([[img_width/2], [0], [1]], dtype=np.float32)
                    neighbor_top = neighbor_top / neighbor_top[2, 0]
                    
                    # 목표: 이웃 이미지의 위가 현재 이미지의 아래에서 target_overlap만큼 위에 위치
                    target_top_y = prev_bottom_y - target_overlap
                    
                    # X축은 호모그래피로 계산한 상대적 위치 사용
                    prev_center = current_transform @ np.array([[img_width/2], [img_height/2], [1]], dtype=np.float32)
                    prev_center = prev_center / prev_center[2, 0]
                    
                    neighbor_center_in_current = H_neighbor_to_current @ np.array([[img_width/2], [img_height/2], [1]], dtype=np.float32)
                    neighbor_center_in_current = neighbor_center_in_current / neighbor_center_in_current[2, 0]
                    
                    relative_offset_x = neighbor_center_in_current[0, 0] - img_width/2
                    target_center_x = prev_center[0, 0] + relative_offset_x
                    
                    neighbor_offset_x = target_center_x - img_width / 2
                    neighbor_offset_y = target_top_y
                    
                elif direction == 'up':
                    # 현재 이미지의 위 가장자리
                    prev_top = current_transform @ np.array([[img_width/2], [0], [1]], dtype=np.float32)
                    prev_top = prev_top / prev_top[2, 0]
                    prev_top_y = prev_top[1, 0]
                    
                    # 이웃 이미지의 아래 가장자리를 현재 이미지 좌표계로 변환
                    neighbor_bottom = H @ np.array([[img_width/2], [img_height], [1]], dtype=np.float32)
                    neighbor_bottom = neighbor_bottom / neighbor_bottom[2, 0]
                    
                    # 목표: 이웃 이미지의 아래가 현재 이미지의 위에서 target_overlap만큼 아래에 위치
                    target_bottom_y = prev_top_y + target_overlap
                    
                    # X축은 호모그래피로 계산한 상대적 위치 사용
                    prev_center = current_transform @ np.array([[img_width/2], [img_height/2], [1]], dtype=np.float32)
                    prev_center = prev_center / prev_center[2, 0]
                    
                    neighbor_center_in_current = H @ np.array([[img_width/2], [img_height/2], [1]], dtype=np.float32)
                    neighbor_center_in_current = neighbor_center_in_current / neighbor_center_in_current[2, 0]
                    
                    relative_offset_x = neighbor_center_in_current[0, 0] - img_width/2
                    target_center_x = prev_center[0, 0] + relative_offset_x
                    
                    neighbor_offset_x = target_center_x - img_width / 2
                    neighbor_offset_y = target_bottom_y - img_height
                
                # 이웃 이미지의 위치 저장 (translation만 사용, 원본 형태 유지)
                image_offsets[neighbor_idx] = (neighbor_offset_x, neighbor_offset_y)
                image_transforms[neighbor_idx] = np.array([
                    [1, 0, neighbor_offset_x],
                    [0, 1, neighbor_offset_y],
                    [0, 0, 1]
                ], dtype=np.float32)
                
                visited.add(neighbor_idx)
                # 거리 순으로 정렬하여 큐에 추가 (가까운 이미지부터 처리)
                queue.append(neighbor_idx)
                # 큐를 거리 순으로 정렬 (현재 이미지와의 거리 기준)
                queue.sort(key=lambda idx: image_distances.get((current_idx, idx), float('inf')))
                
                print(f"  Image {neighbor_idx+1} at ({neighbor_row}, {neighbor_col}): Offset ({neighbor_offset_x:.1f}, {neighbor_offset_y:.1f})")
        
        # 방문하지 않은 이미지가 있으면 기본 위치에 배치
        for idx in range(n):
            if idx not in visited:
                row, col = grid_positions[idx]
                # 그리드 위치에 따라 기본 오프셋 계산
                offset_x = col * img_width * (1 - target_overlap_ratio)
                offset_y = row * img_height * (1 - target_overlap_ratio)
                image_offsets[idx] = (offset_x, offset_y)
                image_transforms[idx] = np.array([
                    [1, 0, offset_x],
                    [0, 1, offset_y],
                    [0, 0, 1]
                ], dtype=np.float32)
                print(f"  Image {idx+1} at ({row}, {col}): Default offset ({offset_x:.1f}, {offset_y:.1f})")
        
        # img_width, img_height는 이미 위에서 계산됨
        
        # 모든 이미지의 위치를 계산하여 캔버스 크기 결정 (원본 형태 유지, translation만 적용)
        all_corners_x = [0, img_width]
        all_corners_y = [0, img_height]
        
        # 각 이미지의 네 모서리 (오프셋만 적용, 변형 없음)
        for i, (offset_x_img, offset_y_img) in image_offsets.items():
            all_corners_x.extend([offset_x_img, offset_x_img + img_width])
            all_corners_y.extend([offset_y_img, offset_y_img + img_height])
        
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
        
        # 각 이미지를 원본 형태 유지하며 배치 (호모그래피로 계산한 위치에 translation만 적용)
        for idx, svg_file in enumerate(svg_files):
            print(f"Placing SVG {idx+1}/{len(svg_files)}...")
            
            # 이미지 오프셋 가져오기 (호모그래피로 계산한 위치)
            if idx in image_offsets:
                img_offset_x, img_offset_y = image_offsets[idx]
                base_offset_x = offset_x + img_offset_x
                base_offset_y = offset_y + img_offset_y
            else:
                # 오프셋이 없으면 기본 위치
                base_offset_x = offset_x
                base_offset_y = offset_y
            
            # 파일명 표시 (옵션)
            if self.show_labels:
                from pathlib import Path
                file_name = Path(svg_file).stem
                
                # 중앙에 큰 텍스트로 표시
                text_elem = ET.SubElement(root, 'text')
                text_elem.set('x', str(base_offset_x + img_width / 2))
                text_elem.set('y', str(base_offset_y + img_height / 2))
                text_elem.set('fill', 'blue')
                text_elem.set('font-size', '120')
                text_elem.set('font-weight', 'bold')
                text_elem.set('font-family', 'Arial, sans-serif')
                text_elem.set('opacity', '0.7')
                text_elem.set('text-anchor', 'middle')  # 중앙 정렬
                text_elem.set('dominant-baseline', 'middle')  # 수직 중앙 정렬
                text_elem.text = f"{file_name}"
            
            # 경계선 표시 (옵션)
            if self.show_borders:
                border_elem = ET.SubElement(root, 'rect')
                border_elem.set('x', str(base_offset_x))
                border_elem.set('y', str(base_offset_y))
                border_elem.set('width', str(img_width))
                border_elem.set('height', str(img_height))
                border_elem.set('fill', 'none')
                border_elem.set('stroke', 'red')
                border_elem.set('stroke-width', '5')
                border_elem.set('opacity', '0.8')
            
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
                
                # Parse and transform path using svg.path for robustness
                from svg.path import parse_path, Line, CubicBezier, QuadraticBezier, Arc, Move, Close
                
                try:
                    path = parse_path(path_d)
                    offset = complex(base_offset_x, base_offset_y)
                    
                    new_d_parts = []
                    
                    for segment in path:
                        # Apply offset to relevant points
                        # Note: segment.start is typically just for reference/Move, 
                        # but modifying it helps consistency in the object model.
                        # What matters for serialization is usually the 'end' and control points, 
                        # except for Move commands.
                        
                        if isinstance(segment, Move):
                            start = segment.start + offset
                            new_d_parts.append(f"M {start.real:.2f} {start.imag:.2f}")
                            
                        elif isinstance(segment, Line):
                            end = segment.end + offset
                            new_d_parts.append(f"L {end.real:.2f} {end.imag:.2f}")
                            
                        elif isinstance(segment, CubicBezier):
                            c1 = segment.control1 + offset
                            c2 = segment.control2 + offset
                            end = segment.end + offset
                            new_d_parts.append(f"C {c1.real:.2f} {c1.imag:.2f} {c2.real:.2f} {c2.imag:.2f} {end.real:.2f} {end.imag:.2f}")
                            
                        elif isinstance(segment, QuadraticBezier):
                            c1 = segment.control1 + offset
                            end = segment.end + offset
                            new_d_parts.append(f"Q {c1.real:.2f} {c1.imag:.2f} {end.real:.2f} {end.imag:.2f}")

                        elif isinstance(segment, Close):
                            new_d_parts.append("Z")
                            
                        elif isinstance(segment, Arc):
                            # Arc is complex (radius, rotation, arc flags stay same, end point moves)
                            end = segment.end + offset
                            # Arguments: rx ry rot large_arc sweep end_x end_y
                            new_d_parts.append(f"A {segment.radius.real:.2f} {segment.radius.imag:.2f} {segment.rotation:.2f} {1 if segment.large_arc else 0} {1 if segment.sweep else 0} {end.real:.2f} {end.imag:.2f}")
                    
                    path_d_transformed = " ".join(new_d_parts)
                    
                except Exception as e:
                    print(f"Error transforming path in {file_name}: {e}")
                    continue

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
                    
                    # 원본 형태 유지, translation만 적용
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
        # Remove all ns0: prefixes
        content = re.sub(r'ns0:', '', content)
        # Fix root tag if needed (though removing ns0: above handles <ns0:svg>)
        # content = re.sub(r'<ns0:svg', '<svg', content) # Redundant now
        # content = re.sub(r'</ns0:svg>', '</svg>', content) # Redundant now
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
    
    def _analyze_relative_position(self, svg1_path: str, svg2_path: str, 
                                   H: np.ndarray, img_width: float, img_height: float) -> Dict:
        """
        두 이미지 간 상대적 위치 분석
        
        Args:
            svg1_path: 첫 번째 SVG
            svg2_path: 두 번째 SVG
            H: 호모그래피 행렬 (svg1 -> svg2)
            img_width: 이미지 너비
            img_height: 이미지 높이
            
        Returns:
            위치 정보 딕셔너리
        """
        # svg1의 중심점을 svg2 좌표계로 변환
        center1 = np.array([[img_width/2], [img_height/2], [1]], dtype=np.float32)
        center1_in_svg2 = H @ center1
        center1_in_svg2 = center1_in_svg2 / center1_in_svg2[2, 0]
        
        # svg2의 중심점
        center2 = np.array([img_width/2, img_height/2])
        
        # 상대적 위치 계산
        offset_x = center1_in_svg2[0, 0] - center2[0]
        offset_y = center1_in_svg2[1, 0] - center2[1]
        
        # 방향 결정
        threshold = min(img_width, img_height) * 0.3
        
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
    
    def _compute_grid_positions(self, n: int, relationships: Dict) -> Dict[int, Tuple[int, int]]:
        """
        2D 그리드 위치 계산
        
        Args:
            n: 이미지 개수
            relationships: {(i, j): {'direction': str, ...}} 딕셔너리
            
        Returns:
            {이미지 인덱스: (row, col)} 딕셔너리
        """
        if n <= 1:
            return {0: (0, 0)}
        
        # 첫 번째 이미지를 (0, 0)에 배치
        positions = {0: (0, 0)}
        visited = {0}
        
        # BFS로 위치 계산
        queue = [0]
        
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
            neighbors.sort(key=lambda x: direction_priority.get(x[1].get('direction', 'unknown'), 4))
            
            for j, rel_info in neighbors:
                direction = rel_info.get('direction', 'unknown')
                
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
        
        # 방문하지 않은 이미지가 있으면 나머지도 배치
        for idx in range(n):
            if idx not in visited:
                # 모든 위치가 차있으면 순차 배치
                max_col = max([p[1] for p in positions.values()]) if positions else 0
                max_row = max([p[0] for p in positions.values()]) if positions else 0
                # 다음 행의 첫 번째 열에 배치
                positions[idx] = (max_row + 1, 0)
                visited.add(idx)
        
        # 정사각형 그리드로 재배치 (계산된 관계 무시하고 단순 그리드)
        import math
        grid_size = int(math.ceil(math.sqrt(n)))
        
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

