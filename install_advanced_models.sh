#!/bin/bash
# 고급 딥러닝 모델 설치 스크립트

echo "Installing advanced deep learning models for image stitching..."

# LightGlue 설치 (SuperGlue 개선 버전, 더 빠르고 정확함)
echo "Installing LightGlue..."
pip install lightglue

# DINOv2는 transformers에 포함되어 있음
echo "DINOv2 is available via transformers package (already installed)"

echo ""
echo "Available models:"
echo "  - LoFTR (already integrated)"
echo "  - DISK (already integrated)"
echo "  - LightGlue (newly installed) - Recommended for better performance"
echo "  - DINOv2 (available via transformers) - Recommended for feature extraction"
echo ""
echo "Usage examples:"
echo "  python stitch_svg_vector.py --method lightglue"
echo "  python stitch_svg_vector.py --method lightglue_disk"

