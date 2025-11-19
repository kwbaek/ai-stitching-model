"""
트랜스포머 기반 SVG 벡터 매칭 모듈
Self-attention과 Cross-attention으로 벡터 경로 간 대응점 찾기
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from svg_vector_analyzer import SVGPathExtractor


class PathEncoder(nn.Module):
    """SVG 경로를 임베딩으로 변환하는 인코더"""
    
    def __init__(self, input_dim: int = 2, hidden_dim: int = 128, 
                 num_layers: int = 2):
        """
        Args:
            input_dim: 입력 차원 (x, y 좌표 = 2)
            hidden_dim: 히든 차원
            num_layers: 레이어 수
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # 좌표 임베딩
        self.coord_embed = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 위치 인코딩
        self.pos_encoder = PositionalEncoding(hidden_dim)
        
        # Transformer 인코더
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: (batch, seq_len, 2) 좌표 텐서
            
        Returns:
            (batch, seq_len, hidden_dim) 임베딩
        """
        # 좌표 임베딩
        x = self.coord_embed(coords)  # (B, L, H)
        
        # 위치 인코딩 추가
        x = self.pos_encoder(x)
        
        # Transformer 인코딩
        x = self.transformer(x)  # (B, L, H)
        
        return x


class PositionalEncoding(nn.Module):
    """사인/코사인 위치 인코딩"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        # 위치 인코딩 계산
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        """
        seq_len = x.size(1)
        x = x + self.pe[:seq_len, :].unsqueeze(0)
        return x


class CrossAttentionMatcher(nn.Module):
    """Cross-attention으로 두 SVG 간 대응점 매칭"""
    
    def __init__(self, hidden_dim: int = 128, num_heads: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # 매칭 점수 계산
        self.match_score = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, embed1: torch.Tensor, embed2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            embed1: (batch, len1, hidden_dim) 첫 번째 SVG 임베딩
            embed2: (batch, len2, hidden_dim) 두 번째 SVG 임베딩
            
        Returns:
            attention_weights: (batch, len1, len2) 어텐션 가중치
            match_scores: (batch, len1, len2) 매칭 점수
        """
        # Cross-attention: embed1을 query, embed2를 key/value로
        attn_output, attn_weights = self.cross_attn(
            query=embed1,
            key=embed2,
            value=embed2,
            need_weights=True
        )
        
        # 매칭 점수 계산
        # embed1의 각 점과 embed2의 각 점 간 유사도
        batch_size, len1, hidden = embed1.shape
        len2 = embed2.shape[1]
        
        # (B, L1, 1, H) x (B, 1, L2, H) -> (B, L1, L2, H)
        embed1_expanded = embed1.unsqueeze(2).expand(-1, -1, len2, -1)
        embed2_expanded = embed2.unsqueeze(1).expand(-1, len1, -1, -1)
        
        # 연결
        combined = torch.cat([embed1_expanded, embed2_expanded], dim=-1)  # (B, L1, L2, 2H)
        
        # 매칭 점수
        match_scores = self.match_score(combined).squeeze(-1)  # (B, L1, L2)
        
        return attn_weights, match_scores


class VectorTransformer(nn.Module):
    """전체 트랜스포머 기반 벡터 매칭 모델"""
    
    def __init__(self, hidden_dim: int = 128, num_encoder_layers: int = 2,
                 num_heads: int = 4):
        super().__init__()
        
        # 경로 인코더
        self.encoder = PathEncoder(
            input_dim=2,
            hidden_dim=hidden_dim,
            num_layers=num_encoder_layers
        )
        
        # Cross-attention 매처
        self.matcher = CrossAttentionMatcher(
            hidden_dim=hidden_dim,
            num_heads=num_heads
        )
    
    def forward(self, coords1: torch.Tensor, coords2: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            coords1: (batch, len1, 2) 첫 번째 SVG 좌표
            coords2: (batch, len2, 2) 두 번째 SVG 좌표
            
        Returns:
            결과 딕셔너리
        """
        # 인코딩
        embed1 = self.encoder(coords1)  # (B, L1, H)
        embed2 = self.encoder(coords2)  # (B, L2, H)
        
        # 매칭
        attn_weights, match_scores = self.matcher(embed1, embed2)
        
        return {
            'embed1': embed1,
            'embed2': embed2,
            'attention_weights': attn_weights,
            'match_scores': match_scores
        }


class TransformerVectorMatcher:
    """트랜스포머 기반 벡터 매칭 인터페이스"""
    
    def __init__(self, model_path: Optional[str] = None, 
                 device: Optional[str] = None,
                 hidden_dim: int = 128):
        """
        Args:
            model_path: 사전 학습된 모델 경로 (없으면 랜덤 초기화)
            device: 'cuda' 또는 'cpu'
            hidden_dim: 히든 차원
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.extractor = SVGPathExtractor()
        
        # 모델 초기화
        self.model = VectorTransformer(hidden_dim=hidden_dim).to(self.device)
        
        # 모델 로드
        if model_path and Path(model_path).exists():
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Loaded model from {model_path}")
        else:
            print("Using randomly initialized model (no pretrained weights)")
        
        self.model.eval()
    
    def preprocess_coords(self, coords_list: List[np.ndarray], 
                         max_points: int = 1000) -> torch.Tensor:
        """
        좌표 리스트를 텐서로 변환 및 정규화
        
        Args:
            coords_list: 좌표 배열 리스트
            max_points: 최대 포인트 수
            
        Returns:
            (1, num_points, 2) 텐서
        """
        # 모든 좌표 합치기
        all_coords = []
        for coords in coords_list:
            if len(coords) > 0:
                all_coords.append(coords)
        
        if not all_coords:
            # 빈 경우 더미 데이터
            return torch.zeros(1, 1, 2, device=self.device)
        
        coords = np.vstack(all_coords)
        
        # 샘플링 (너무 많으면)
        if len(coords) > max_points:
            indices = np.linspace(0, len(coords) - 1, max_points, dtype=int)
            coords = coords[indices]
        
        # 정규화 (0-1 범위로)
        min_vals = coords.min(axis=0)
        max_vals = coords.max(axis=0)
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1  # 0으로 나누기 방지
        
        coords_norm = (coords - min_vals) / range_vals
        
        # 텐서 변환
        coords_tensor = torch.from_numpy(coords_norm).float().unsqueeze(0)
        coords_tensor = coords_tensor.to(self.device)
        
        return coords_tensor
    
    def extract_svg_coords(self, svg_path: str) -> List[np.ndarray]:
        """SVG 파일에서 좌표 추출"""
        paths = self.extractor.extract_paths_from_svg(svg_path)
        
        coords_list = []
        for path_info in paths:
            coords = path_info.get('coords', [])
            if coords:
                coords_list.append(np.array(coords))
        
        return coords_list
    
    def match_vectors(self, svg1_path: str, svg2_path: str,
                     match_threshold: float = 0.5) -> Dict:
        """
        두 SVG 파일 간 벡터 매칭
        
        Args:
            svg1_path: 첫 번째 SVG 파일
            svg2_path: 두 번째 SVG 파일
            match_threshold: 매칭 임계값
            
        Returns:
            매칭 결과 딕셔너리
        """
        # 좌표 추출
        coords1_list = self.extract_svg_coords(svg1_path)
        coords2_list = self.extract_svg_coords(svg2_path)
        
        if not coords1_list or not coords2_list:
            return {
                'keypoints0': np.array([]),
                'keypoints1': np.array([]),
                'confidence': np.array([]),
                'num_matches': 0
            }
        
        # 전처리
        coords1_tensor = self.preprocess_coords(coords1_list)
        coords2_tensor = self.preprocess_coords(coords2_list)
        
        # 추론
        with torch.no_grad():
            results = self.model(coords1_tensor, coords2_tensor)
        
        # 매칭 점수
        match_scores = results['match_scores'][0].cpu().numpy()  # (L1, L2)
        
        # 임계값 이상인 매칭만 선택
        matches = np.argwhere(match_scores > match_threshold)
        
        if len(matches) == 0:
            return {
                'keypoints0': np.array([]),
                'keypoints1': np.array([]),
                'confidence': np.array([]),
                'num_matches': 0
            }
        
        # 원본 좌표 복원
        coords1_np = coords1_tensor[0].cpu().numpy()
        coords2_np = coords2_tensor[0].cpu().numpy()
        
        # 매칭 포인트 추출
        idx1 = matches[:, 0]
        idx2 = matches[:, 1]
        
        keypoints0 = coords1_np[idx1]
        keypoints1 = coords2_np[idx2]
        confidence = match_scores[idx1, idx2]
        
        return {
            'keypoints0': keypoints0,
            'keypoints1': keypoints1,
            'confidence': confidence,
            'num_matches': len(matches)
        }
