# 모델별 테스트 결과

## 테스트 환경
- 이미지 수: 10개
- 테스트 날짜: 2024
- GPU: CUDA 사용 가능

## 결과 요약

| 모델 | 상태 | 처리 시간 | 파일 크기 | 캔버스 크기 | 특징 |
|------|------|----------|----------|------------|------|
| **Basic Vector Matching** | ✅ 성공 | 2.57초 | 1.74 MB | 18665 x 16469 | 기본 거리 기반 매칭 |
| **Transformer Vector Matching** | ✅ 성공 | 2.55초 | 1.75 MB | 16517 x 17619 | Self-attention 기반 |
| **GNN Vector Matching** | ✅ 성공 | **2.35초** | **1.72 MB** | **15590 x 16495** | Graph Neural Network |

## 상세 분석

### 1. Basic Vector Matching
- **처리 시간**: 2.57초
- **파일 크기**: 1.74 MB
- **캔버스 크기**: 18665 x 16469 (가장 큼)
- **특징**: 
  - 가장 기본적인 방법
  - 정규화된 좌표 거리 기반 매칭
  - 안정적이고 빠름

### 2. Transformer Vector Matching
- **처리 시간**: 2.55초
- **파일 크기**: 1.75 MB
- **캔버스 크기**: 16517 x 17619
- **특징**:
  - Self-attention과 Cross-attention 사용
  - 벡터 경로 간 대응점 찾기
  - 더 정교한 매칭 가능
  - 현재는 랜덤 초기화 모델 사용 (사전 학습 가중치 없음)

### 3. GNN Vector Matching ⭐ 최고 성능
- **처리 시간**: **2.35초** (가장 빠름)
- **파일 크기**: **1.72 MB** (가장 작음)
- **캔버스 크기**: **15590 x 16495** (가장 컴팩트)
- **특징**:
  - SVG 경로를 그래프로 표현
  - GAT (Graph Attention Network) 사용
  - 구조적 관계를 고려한 매칭
  - 가장 효율적인 결과

## 결론

**GNN Vector Matching**이 가장 우수한 성능을 보였습니다:
- ✅ 가장 빠른 처리 속도 (2.35초)
- ✅ 가장 작은 파일 크기 (1.72 MB)
- ✅ 가장 컴팩트한 캔버스 크기

모든 모델이 성공적으로 스티칭을 완료했으며, 각 모델의 특성에 따라 약간씩 다른 결과를 생성했습니다.

## 생성된 파일

1. `panorama_basic_vector_matching.svg` - 기본 벡터 매칭 결과
2. `panorama_transformer_vector_matching.svg` - Transformer 기반 결과
3. `panorama_gnn_vector_matching.svg` - GNN 기반 결과 (추천)

## 사용 방법

```bash
# 모든 모델 테스트
python3 test_all_models.py

# 개별 모델 사용
python3 stitch_svg_vector.py --input_dir m2/ --output panorama_basic.svg
python3 stitch_svg_vector.py --input_dir m2/ --output panorama_transformer.svg --use_transformer
python3 stitch_svg_vector.py --input_dir m2/ --output panorama_gnn.svg --use_gnn
```

