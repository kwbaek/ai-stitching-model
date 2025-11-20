# 고급 딥러닝 모델 통합 가이드

SVG 벡터 이미지 스티칭을 위한 최신 딥러닝/트랜스포머 모델 통합 방법

## 현재 사용 중인 모델

1. **LoFTR** (Detector-Free Local Feature Matching with Transformers)
   - Transformer 기반 특징점 매칭
   - Detector-free 방식으로 밀집 매칭 제공
   - 현재 코드베이스에 통합됨

2. **DISK** (Differentiable Inlier Scoring for Keypoints)
   - 학습 가능한 특징점 검출 및 매칭
   - 현재 코드베이스에 통합됨

3. **Custom Transformer/GNN** 벡터 매칭
   - SVG 경로 좌표 직접 처리
   - 현재 코드베이스에 통합됨

## 추천 추가 모델

### 1. LightGlue (SuperGlue 개선 버전) ⭐ 추천

**특징:**
- SuperGlue보다 2-3배 빠르고 더 정확함
- Adaptive pruning으로 효율성 향상
- Transformer 기반 attention 매칭

**설치:**
```bash
pip install lightglue
```

**GitHub:** https://github.com/cvg/LightGlue

**사용 예시:**
```python
from lightglue import LightGlue, SuperPoint, DISK
from lightglue.utils import load_image, rbd

extractor = SuperPoint(max_num_keypoints=2048).eval().cuda()
matcher = LightGlue(features='superpoint').eval().cuda()

image0 = load_image('path/to/image0.jpg')
image1 = load_image('path/to/image1.jpg')

feats0 = extractor.extract(image0.cuda())
feats1 = extractor.extract(image1.cuda())
matches01 = matcher({'image0': feats0, 'image1': feats1})
```

### 2. DINOv2 (Self-Supervised Vision Transformer) ⭐ 추천

**특징:**
- Meta AI의 최신 self-supervised vision transformer
- 강력한 특징 추출 능력
- 다양한 이미지에 일반화 성능 우수

**설치:**
```bash
pip install transformers
```

**GitHub:** https://github.com/facebookresearch/dinov2

**사용 예시:**
```python
import torch
from transformers import AutoImageProcessor, AutoModel

processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
model = AutoModel.from_pretrained('facebook/dinov2-base').cuda()

# 이미지 특징 추출
inputs = processor(images=image, return_tensors="pt").to('cuda')
with torch.no_grad():
    outputs = model(**inputs)
    features = outputs.last_hidden_state
```

### 3. ASpanFormer (Attention-based Span Matching)

**특징:**
- Attention 기반 span matching
- 긴 거리 의존성 학습에 강함
- 파노라마 스티칭에 적합

**GitHub:** https://github.com/apple/ml-aspanformer

### 4. RoMa (Robust Matching)

**특징:**
- Robust한 매칭 성능
- 다양한 조건에서 안정적
- Transformer 기반

**GitHub:** https://github.com/Parskatt/RoMa

### 5. DeDoDe (Detect, Don't Describe)

**특징:**
- 최신 detector-free 매칭
- LoFTR와 유사하지만 개선된 성능
- Transformer 기반

**GitHub:** https://github.com/Parskatt/DeDoDe

## 통합 우선순위

1. **LightGlue** - 가장 빠르고 정확한 매칭 성능
2. **DINOv2** - 강력한 특징 추출로 매칭 품질 향상
3. **DeDoDe** - LoFTR의 개선 버전

## 통합 방법

각 모델을 `feature_matcher.py`에 추가하여 선택적으로 사용할 수 있도록 구현

