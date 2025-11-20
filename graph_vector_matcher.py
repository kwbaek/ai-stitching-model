"""
Graph Neural Network 기반 SVG 벡터 매칭 모델
SVG 경로를 그래프로 표현하고 GNN으로 매칭
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path

try:
    from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool
    from torch_geometric.data import Data, Batch
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    print("Warning: torch-geometric not available. Install with: pip install torch-geometric")

from svg_vector_analyzer import SVGPathExtractor


if not TORCH_GEOMETRIC_AVAILABLE:
    # torch-geometric이 없으면 더미 클래스
    class PathGraphBuilder:
        def __init__(self, *args, **kwargs):
            raise ImportError("torch-geometric is required. Install with: pip install torch-geometric")
    
    class GraphVectorMatcher:
        def __init__(self, *args, **kwargs):
            raise ImportError("torch-geometric is required. Install with: pip install torch-geometric")
else:
    class PathGraphBuilder:
        """SVG 경로를 그래프로 변환"""
        
        def __init__(self, max_nodes: int = 1000):
            self.max_nodes = max_nodes
            self.extractor = SVGPathExtractor()
        
        def build_graph(self, svg_path: str) -> Optional[Data]:
            """
            SVG 파일을 그래프로 변환
            
            Args:
                svg_path: SVG 파일 경로
                
            Returns:
                PyTorch Geometric Data 객체
            """
            paths = self.extractor.extract_paths_from_svg(svg_path)
            
            if not paths:
                return None
            
            # 모든 좌표 수집
            all_coords = []
            node_features = []
            
            for path_info in paths:
                coords = path_info.get('coords', [])
                if len(coords) > 0:
                    coords_array = np.array(coords)
                    all_coords.extend(coords_array)
                    
                    # 노드 특징: 좌표, 경로 길이, 중심점
                    for coord in coords_array:
                        # 특징: [x, y, path_length, center_x, center_y]
                        path_center = coords_array.mean(axis=0)
                        path_length = np.linalg.norm(coords_array[-1] - coords_array[0])
                        features = np.array([
                            coord[0], coord[1],
                            path_length,
                            path_center[0], path_center[1]
                        ])
                        node_features.append(features)
            
            if len(all_coords) == 0:
                return None
            
            # 샘플링 (너무 많으면)
            if len(all_coords) > self.max_nodes:
                indices = np.linspace(0, len(all_coords) - 1, self.max_nodes, dtype=int)
                all_coords = [all_coords[i] for i in indices]
                node_features = [node_features[i] for i in indices]
            
            # 정규화
            coords_array = np.array(all_coords)
            min_vals = coords_array.min(axis=0)
            max_vals = coords_array.max(axis=0)
            range_vals = max_vals - min_vals
            range_vals[range_vals == 0] = 1
            
            coords_norm = (coords_array - min_vals) / range_vals
            
            # 노드 특징 정규화
            features_array = np.array(node_features)
            features_min = features_array.min(axis=0)
            features_max = features_array.max(axis=0)
            features_range = features_max - features_min
            features_range[features_range == 0] = 1
            features_norm = (features_array - features_min) / features_range
            
            # 엣지 생성 (k-NN 그래프)
            from sklearn.neighbors import NearestNeighbors
            k = min(5, len(coords_norm) - 1)
            if k > 0:
                nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='ball_tree').fit(coords_norm)
                distances, indices = nbrs.kneighbors(coords_norm)
                
                # 엣지 리스트 생성
                edge_index = []
                for i, neighbors in enumerate(indices):
                    for j in neighbors[1:]:  # 자기 자신 제외
                        edge_index.append([i, j])
                        edge_index.append([j, i])  # 양방향
                
                edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)
            
            # PyTorch Geometric Data 객체 생성
            data = Data(
                x=torch.tensor(features_norm, dtype=torch.float),
                edge_index=edge_index,
                pos=torch.tensor(coords_norm, dtype=torch.float)
            )
            
            return data

    class GNNEncoder(nn.Module):
        """Graph Neural Network 인코더"""
        
        def __init__(self, input_dim: int = 5, hidden_dim: int = 128, 
                     num_layers: int = 3, use_gat: bool = True):
            super().__init__()
            self.hidden_dim = hidden_dim
            self.num_layers = num_layers
            
            # 입력 레이어
            self.input_linear = nn.Linear(input_dim, hidden_dim)
            
            # GNN 레이어
            self.convs = nn.ModuleList()
            for i in range(num_layers):
                if use_gat:
                    conv = GATConv(
                        hidden_dim if i > 0 else hidden_dim,
                        hidden_dim,
                        heads=4,
                        concat=False,
                        dropout=0.1
                    )
                else:
                    conv = GCNConv(
                        hidden_dim if i > 0 else hidden_dim,
                        hidden_dim
                    )
                self.convs.append(conv)
            
            # 배치 정규화
            self.batch_norms = nn.ModuleList([
                nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)
            ])
        
        def forward(self, data: Data) -> torch.Tensor:
            """
            Args:
                data: PyTorch Geometric Data 객체
                
            Returns:
                (num_nodes, hidden_dim) 노드 임베딩
            """
            x = data.x
            edge_index = data.edge_index
            
            # 입력 변환
            x = self.input_linear(x)
            
            # GNN 레이어
            for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
                x_new = conv(x, edge_index)
                x_new = bn(x_new)
                x_new = F.relu(x_new)
                
                # Residual connection
                if i > 0:
                    x = x + x_new
                else:
                    x = x_new
            
            return x

    class GraphVectorMatcher:
        """GNN 기반 벡터 매칭"""
        
        def __init__(self, model_path: Optional[str] = None,
                     device: Optional[str] = None,
                     hidden_dim: int = 128):
            self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
            self.graph_builder = PathGraphBuilder()
            
            # 모델 초기화
            self.encoder = GNNEncoder(
                input_dim=5,
                hidden_dim=hidden_dim,
                num_layers=3,
                use_gat=True
            ).to(self.device)
            
            # 매칭 네트워크
            self.match_network = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid()
            ).to(self.device)
            
            # 모델 로드
            if model_path and Path(model_path).exists():
                checkpoint = torch.load(model_path, map_location=self.device)
                self.encoder.load_state_dict(checkpoint.get('encoder', {}))
                self.match_network.load_state_dict(checkpoint.get('match_network', {}))
                print(f"Loaded model from {model_path}")
            else:
                print("Using randomly initialized GNN model")
            
            self.encoder.eval()
            self.match_network.eval()
        
        def encode_graph(self, graph: Data) -> torch.Tensor:
            """그래프를 임베딩으로 인코딩"""
            graph = graph.to(self.device)
            node_embeddings = self.encoder(graph)
            
            # 그래프 레벨 임베딩 (평균 풀링)
            graph_embedding = global_mean_pool(node_embeddings, batch=None)
            
            return graph_embedding, node_embeddings
        
        def match_graphs(self, svg1_path: str, svg2_path: str,
                        match_threshold: float = 0.5) -> Dict:
            """
            두 SVG 파일 간 그래프 매칭
            
            Args:
                svg1_path: 첫 번째 SVG 파일
                svg2_path: 두 번째 SVG 파일
                match_threshold: 매칭 임계값
                
            Returns:
                매칭 결과 딕셔너리
            """
            # 그래프 생성
            graph1 = self.graph_builder.build_graph(svg1_path)
            graph2 = self.graph_builder.build_graph(svg2_path)
            
            if graph1 is None or graph2 is None:
                return {
                    'keypoints0': np.array([]),
                    'keypoints1': np.array([]),
                    'confidence': np.array([]),
                    'num_matches': 0
                }
            
            # 인코딩
            with torch.no_grad():
                embed1, nodes1 = self.encode_graph(graph1)
                embed2, nodes2 = self.encode_graph(graph2)
                
                # 노드 간 매칭 점수 계산
                # (N1, H) x (N2, H) -> (N1, N2)
                nodes1_expanded = nodes1.unsqueeze(1)  # (N1, 1, H)
                nodes2_expanded = nodes2.unsqueeze(0)  # (1, N2, H)
                
                # 연결
                combined = torch.cat([nodes1_expanded.expand(-1, nodes2.size(0), -1),
                                     nodes2_expanded.expand(nodes1.size(0), -1, -1)], dim=-1)
                
                # 매칭 점수
                match_scores = self.match_network(combined).squeeze(-1)  # (N1, N2)
                match_scores_np = match_scores.cpu().numpy()
            
            # 임계값 이상인 매칭만 선택
            matches = np.argwhere(match_scores_np > match_threshold)
            
            if len(matches) == 0:
                return {
                    'keypoints0': np.array([]),
                    'keypoints1': np.array([]),
                    'confidence': np.array([]),
                    'num_matches': 0
                }
            
            # 원본 좌표 복원
            pos1 = graph1.pos.cpu().numpy()
            pos2 = graph2.pos.cpu().numpy()
            
            idx1 = matches[:, 0]
            idx2 = matches[:, 1]
            
            keypoints0 = pos1[idx1]
            keypoints1 = pos2[idx2]
            confidence = match_scores_np[idx1, idx2]
            
            return {
                'keypoints0': keypoints0,
                'keypoints1': keypoints1,
                'confidence': confidence,
                'num_matches': len(matches)
            }

