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
- SVG 변환 행렬 직접 적용
- 벡터 그래픽 품질 유지
- 파노라마 SVG 생성

## 설치

```bash
pip install -r requirements.txt
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

stitcher = SVGVectorStitcher()

# SVG 파일 목록
svg_files = ['m2/label0001.svg', 'm2/label0002.svg', ...]

# 벡터 파노라마 생성
stitcher.create_panorama_svg(
    svg_files=svg_files,
    output_path='panorama.svg',
    max_images=10
)
```

## 모델 및 방법

### 딥러닝 기반
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
├── stitch_panorama.py            # 래스터 스티칭 실행 스크립트
├── stitch_svg_vector.py          # 벡터 스티칭 실행 스크립트 ⭐
├── example_usage.py              # 사용 예제
├── requirements.txt              # 의존성
└── README.md                     # 이 파일
```

## 주의사항

- GPU가 있으면 자동으로 사용됩니다 (CUDA)
- 많은 이미지를 처리할 때는 메모리 사용량에 주의하세요
- 첫 실행 시 딥러닝 모델이 자동으로 다운로드됩니다

