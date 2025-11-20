# 딥러닝 모델 비교 및 추천

SVG 벡터 이미지 스티칭을 위한 딥러닝/트랜스포머 모델 성능 비교

## 모델 성능 비교

### 1. LightGlue ⭐⭐⭐ 추천

**장점:**
- SuperGlue보다 2-3배 빠름
- 더 정확한 매칭 성능
- Adaptive pruning으로 효율성 향상
- Transformer 기반 attention 매칭

**단점:**
- 추가 설치 필요 (`pip install lightglue`)

**사용 시나리오:**
- 빠르고 정확한 매칭이 필요한 경우
- 대량 이미지 처리
- 실시간 스티칭

**설치:**
```bash
pip install lightglue
```

**사용:**
```bash
python stitch_svg_vector.py --input_dir m2/ --output panorama.svg --method lightglue
```

### 2. LoFTR ⭐⭐ (현재 기본값)

**장점:**
- Detector-free 방식
- 밀집 매칭 제공
- 이미 통합됨 (추가 설치 불필요)

**단점:**
- LightGlue보다 느림
- 메모리 사용량이 큼

**사용 시나리오:**
- 기본적인 스티칭 작업
- 추가 설치 없이 사용하고 싶은 경우

### 3. DINOv2 ⭐⭐

**장점:**
- Meta AI의 최신 self-supervised vision transformer
- 강력한 특징 추출 능력
- 다양한 이미지에 일반화 성능 우수
- transformers 패키지에 포함 (추가 설치 불필요)

**단점:**
- 패치 기반 매칭 (더 정교한 구현 필요)
- 현재 구현은 간단한 버전

**사용 시나리오:**
- 강력한 특징 추출이 필요한 경우
- 다양한 이미지 타입 처리

**사용:**
```bash
python stitch_svg_vector.py --input_dir m2/ --output panorama.svg --method dinov2
```

### 4. DISK

**장점:**
- 학습 가능한 특징점 검출
- 이미 통합됨

**단점:**
- LoFTR보다 성능이 낮을 수 있음

**사용 시나리오:**
- 특정 특징점 검출이 필요한 경우

## 추천 순서

1. **LightGlue** - 가장 빠르고 정확함 (추천)
2. **LoFTR** - 기본값, 안정적
3. **DINOv2** - 강력한 특징 추출 (실험적)
4. **DISK** - 대안

## 벡터 기반 매칭 (SVG 직접 처리)

### GNN (Graph Neural Network) ⭐⭐⭐
- SVG 경로를 그래프로 표현
- 구조적 관계를 고려한 매칭
- 최고 정확도

### Transformer ⭐⭐
- Self-attention과 Cross-attention
- 벡터 경로 매칭
- 정교한 매칭 가능

### 기본 거리 기반
- 빠른 처리 속도
- 간단하고 안정적

## 성능 최적화 팁

1. **GPU 사용**: CUDA가 가능하면 자동으로 GPU 사용
2. **배치 처리**: 여러 이미지를 한 번에 처리
3. **모델 선택**: LightGlue가 가장 빠르고 정확함
4. **이미지 크기**: 너무 큰 이미지는 리사이즈 고려

## 참고 자료

- LightGlue: https://github.com/cvg/LightGlue
- LoFTR: https://github.com/zju3dv/LoFTR
- DINOv2: https://github.com/facebookresearch/dinov2
- DISK: https://github.com/cvlab-epfl/disk

