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
### 4. End-to-End Pipeline & Visualizations 🆕
- **Full Pipeline**: PNG -> Vectorize -> Multip-GPU Stitching -> GDSII Export
- **Real-time Monitoring**: Web-based UI to track progress and visualize results

## 설치 및 실행

### 기본 설치
```bash
sudo apt-get install potrace  # For vectorization
pip install -r requirements.txt
```

### 웹 서버 실행 (필수)
UI 시각화 및 파일 업로드를 위해 Flask 서버를 실행해야 합니다.

**서버 시작 (백그라운드 실행):**
```bash
python3 server.py > server.log 2>&1 &
```

**서버 중지:**
```bash
pkill -f server.py
# 또는
pkill -f python3
```

- **Dashboard UI**: [http://localhost:8000](http://localhost:8000)
- **Monitoring UI**: [http://localhost:8000/progress.html](http://localhost:8000/progress.html)

### 전체 파이프라인 실행
PNG 이미지부터 시작하여 GDS 생성까지 한번에 실행:
```bash
python3 run_stitching.py --vectorize --show-labels --show-borders --limit 324
```
- `--vectorize`: PNG -> SVG 변환 수행
- `--show-labels`: 결과에 파일명 라벨 표시
- `--show-borders`: 결과에 타일 경계선 표시
- `--limit`: 처리할 파일 수 제한 (전체는 324)

### 파이프라인 단계별 설명
1. **Vectorization**: `utils/vectorize_images.py`를 사용해 PNG를 SVG로 변환 (`potrace` 사용)
2. **Stitching**: `SVGVectorStitcher`가 Multi-GPU를 사용하여 병렬 매칭 수행
3. **GDS Export**: `utils/svg_to_gds.py`를 사용해 최종 SVG를 GDSII 포맷으로 변환

---

## 기존 기능 및 상세 옵션

### 고급 딥러닝 모델 설치 (선택적)
```bash
# LightGlue 설치 (추천)
pip install lightglue
```

### 개별 스크립트 사용법

#### 벡터 스티칭 (기존)
```bash
python stitch_svg_vector.py --input_dir m2/ --output panorama.svg --max_images 10
```

#### GitHub 저장소 다운로드
```bash
python download_svg_from_github.py --repo https://github.com/emsec/unsupervised-ic-sem-segmentation-extended --max_files 50
```

## 모델 및 방법

### 딥러닝 특징점 매칭
1. **LoFTR** (기본): Transformer 기반 밀집 매칭
2. **LightGlue** (추천): SuperGlue의 개선 버전, 빠르고 정확함

### 프로젝트 구조
```
ai-stitching-model/
├── run_stitching.py              # 메인 실행 스크립트 (파이프라인)
├── vectorize_images.py           # PNG → SVG 벡터화 모듈
├── svg_to_gds.py                 # SVG → GDSII 변환 모듈
├── svg_vector_stitcher.py        # 핵심 스티칭 로직 (Multi-GPU 지원)
├── progress.html                 # 실시간 진행상황 모니터링 UI
├── visualize_panorama.html       # 결과 뷰어 UI
└── ...
```

## 주의사항
- **GPU 사용**: 가능한 경우 Multi-GPU를 자동으로 활용합니다.
- **메모리**: 대량의 이미지를 처리할 때 시스템 메모리를 확인하세요.

## MCP Server Integration (Optional)

This pipeline supports the Model Context Protocol (MCP), allowing AI agents to interact with it directly.

### Features
*   **Trigger Pipeline**: Start stitching jobs from chat.
*   **Monitor Status**: Check stage and progress percentage.
*   **Retrieve Results**: Get the final SVG code.

### Configuration
Add this to your MCP settings file (e.g., `claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "ai-stitching": {
      "command": "/app/data/ai-stitching-model/run_mcp_server.sh",
      "args": []
    }
  }
}
```

### Available Tools
*   `start_stitching`: Runs the full pipeline.
*   `get_stitching_status`: Returns current progress JSON.
*   `get_result_preview`: Returns the final SVG (converted to PNG for reliable viewing).
*   `upload_source_image`: Uploads IC chip images for processing.
*   `get_logs`: Retrieves pipeline or server logs.
*   `monitor_pipeline`: Waits for pipeline completion (notifications).
*   `run_full_workflow`: Runs process using files in `dataset/sems/manual`, waits, and returns a text summary with download links (GDS/SVG).
*   `get_vector_preview`: Shows vectorized SVG files (converted to PNG) before stitching.

> **Note**: When running via MCP, the pipeline's console output is suppressed to maintain protocol integrity. Use `get_stitching_status` to monitor progress.

