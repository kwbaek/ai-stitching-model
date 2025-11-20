# AI 기반 SVG 파노라마 스티칭

딥러닝과 트랜스포머 모델을 활용하여 SVG 벡터 이미지를 파노라마 사진처럼 이어붙이는 프로젝트입니다.

## 주요 기능

### 1. 래스터 기반 스티칭
- SVG 벡터 이미지를 래스터 이미지로 변환
- 딥러닝 기반 특징점 추출 및 매칭 (LoFTR, DISK)
- 전통적인 방법 지원 (SIFT, ORB)
- 자동 이미지 정렬 및 호모그래피 계산
- 파노라마 이미지 생성 및 블렌딩

### 2. 벡터 기반 스티칭 (권장) ⭐
- **SVG 경로 좌표 직접 추출 및 분석**
- 벡터 특징점 기반 매칭 (정확도 향상)
- **Graph Neural Network (GNN) 기반 매칭** 🆕
- **Transformer 기반 매칭** 🆕
- SVG 변환 행렬 직접 적용
- 벡터 그래픽 품질 유지
- **10% overlap 자동 조정** 🆕
- **정사각형 그리드 레이아웃** 🆕
- 파노라마 SVG 생성

### 3. GitHub 저장소 통합 🆕
- GitHub 저장소에서 SVG 파일 자동 다운로드
- 대량 SVG 파일 일괄 처리
- 자동 정렬 및 스티칭

## 설치

```bash
pip install -r requirements.txt
```

## 설치

### 기본 설치
```bash
pip install -r requirements.txt
```

### 고급 딥러닝 모델 설치 (선택적, 더 나은 성능)
```bash
# LightGlue 설치 (추천)
pip install lightglue

# 또는 설치 스크립트 실행
./install_advanced_models.sh
```

## 사용법

### 기본 사용 (LoFTR 트랜스포머 모델)

```bash
python stitch_panorama.py --input_dir m2/ --output output_panorama.png --method loftr
```

### 전통적인 방법 (SIFT)

```bash
python stitch_panorama.py --input_dir m2/ --output output_panorama.png --method sift --no_deep_learning
```

### 제한된 이미지 수로 테스트

```bash
python stitch_panorama.py --input_dir m2/ --output output_panorama.png --max_images 10
```

### 벡터 기반 스티칭 (권장)

```bash
python stitch_svg_vector.py --input_dir m2/ --output panorama.svg --max_images 10
```

### GitHub 저장소에서 SVG 다운로드 및 스티칭 🆕

```bash
# GitHub 저장소에서 SVG 다운로드 및 스티칭 (기본 매칭)
python stitch_github_svgs.py --download --max_images 20 --output panorama_github.svg

# Graph Neural Network 기반 매칭 사용
python stitch_github_svgs.py --download --use_gnn --max_images 20 --output panorama_gnn.svg

# Transformer 기반 매칭 사용
python stitch_github_svgs.py --download --use_transformer --max_images 20 --output panorama_transformer.svg

# 10% overlap 명시적 설정
python stitch_github_svgs.py --download --overlap 0.1 --max_images 20
```

### SVG 파일만 다운로드

```bash
python download_svg_from_github.py --repo https://github.com/emsec/unsupervised-ic-sem-segmentation-extended --max_files 50
```

### Python 코드로 사용

#### 래스터 기반
```python
from panorama_stitcher import PanoramaStitcher

# LoFTR 트랜스포머 모델 사용
stitcher = PanoramaStitcher(
    feature_method='loftr',
    use_deep_learning=True,
    output_size=(2048, 1768)
)

# 파노라마 생성
panorama = stitcher.stitch_from_svg_dir(
    svg_dir='m2/',
    output_path='output_panorama.png',
    max_images=10
)
```

#### 벡터 기반 (권장)
```python
from svg_vector_stitcher import SVGVectorStitcher

# 기본 매칭
stitcher = SVGVectorStitcher()

# GNN 기반 매칭 (더 정확함)
stitcher = SVGVectorStitcher(use_gnn=True)

# Transformer 기반 매칭
stitcher = SVGVectorStitcher(use_transformer=True)

# SVG 파일 목록
svg_files = ['m2/label0001.svg', 'm2/label0002.svg', ...]

# 벡터 파노라마 생성 (10% overlap 자동 조정)
stitcher.create_panorama_svg(
    svg_files=svg_files,
    output_path='panorama.svg',
    max_images=10
)
```

## 모델 및 방법

### 딥러닝 특징점 매칭 모델 (이미지 스티칭용)

#### 현재 사용 가능한 모델:
1. **LoFTR** (Detector-Free Local Feature Matching with Transformers) ⭐ 기본
   - Transformer 기반 밀집 매칭
   - Detector-free 방식
   
2. **DISK** (Differentiable Inlier Scoring for Keypoints)
   - 학습 가능한 특징점 검출 및 매칭

3. **LightGlue** 🆕 추천
   - SuperGlue의 개선 버전
   - 2-3배 빠르고 더 정확한 매칭
   - 설치: `pip install lightglue`
   - 사용: `--method lightglue` 또는 `--method lightglue_disk`

4. **DINOv2** 🆕 추천
   - Meta AI의 self-supervised vision transformer
   - 강력한 특징 추출 능력
   - transformers 패키지에 포함됨
   - 사용: `--method dinov2`

### 벡터 기반 매칭 (SVG 스티칭용) 🆕
- **Graph Neural Network (GNN)**: SVG 경로를 그래프로 표현하여 매칭 (최고 정확도)
- **Transformer**: Self-attention과 Cross-attention으로 벡터 경로 매칭
- **기본 거리 기반**: 정규화된 좌표 거리 기반 매칭

### 래스터 기반 매칭
- **LoFTR**: Transformer 기반 특징 매칭 (권장)
- **DISK**: 딥러닝 특징점 검출기

### 전통적인 방법
- **SIFT**: Scale-Invariant Feature Transform
- **ORB**: Oriented FAST and Rotated BRIEF

## 프로젝트 구조

```
ai-stitching-model/
├── svg_converter.py              # SVG → 래스터 변환
├── feature_matcher.py            # 래스터 특징점 매칭
├── image_aligner.py              # 이미지 정렬 및 호모그래피 계산
├── panorama_stitcher.py           # 래스터 기반 스티칭 파이프라인
├── svg_vector_analyzer.py        # SVG 벡터 데이터 분석 ⭐
├── svg_vector_stitcher.py        # 벡터 기반 스티칭 파이프라인 ⭐
├── transformer_vector_matcher.py # Transformer 기반 벡터 매칭 🆕
├── graph_vector_matcher.py       # GNN 기반 벡터 매칭 🆕
├── download_svg_from_github.py   # GitHub 저장소에서 SVG 다운로드 🆕
├── stitch_panorama.py            # 래스터 스티칭 실행 스크립트
├── stitch_svg_vector.py          # 벡터 스티칭 실행 스크립트 ⭐
├── stitch_github_svgs.py         # GitHub SVG 스티칭 스크립트 🆕
├── example_usage.py              # 사용 예제
├── requirements.txt              # 의존성
└── README.md                     # 이 파일
```

## 주요 특징

### 10% Overlap 자동 조정
- 각 이미지 간 약 10%의 overlap을 자동으로 유지
- 호모그래피 기반 정렬 후 overlap 비율 자동 조정
- 위성 타일 스티칭과 유사한 방식

### 정사각형 그리드 레이아웃
- 이미지들을 정사각형 그리드로 배치
- 각 이미지는 원본 그대로 유지 (변형 없음)
- 위치만 오프셋으로 조정

### 벡터 품질 유지
- SVG 벡터 데이터를 직접 처리하여 품질 손실 없음
- PNG 변환 없이 벡터 좌표 직접 조작
- 확대/축소 시에도 선명한 결과

## 주의사항

- GPU가 있으면 자동으로 사용됩니다 (CUDA)
- 많은 이미지를 처리할 때는 메모리 사용량에 주의하세요
- 첫 실행 시 딥러닝 모델이 자동으로 다운로드됩니다
- **GNN 매칭 사용 시 `torch-geometric` 설치 필요**:
  ```bash
  pip install torch-geometric
  ```

