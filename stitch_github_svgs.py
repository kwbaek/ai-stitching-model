"""
GitHub 저장소의 SVG 파일들을 스티칭하는 메인 스크립트
"""
import argparse
from pathlib import Path
from download_svg_from_github import main as download_svgs
from svg_vector_stitcher import SVGVectorStitcher


def main():
    parser = argparse.ArgumentParser(
        description="Download and stitch SVG files from GitHub repository"
    )
    parser.add_argument(
        "--repo",
        type=str,
        default="https://github.com/emsec/unsupervised-ic-sem-segmentation-extended",
        help="GitHub repository URL"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="input_svgs",
        help="Input directory for SVG files (if not downloading)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="panorama_github.svg",
        help="Output SVG file path"
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Maximum number of images to process"
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download SVG files from GitHub repository"
    )
    parser.add_argument(
        "--use_gnn",
        action="store_true",
        help="Use Graph Neural Network for matching"
    )
    parser.add_argument(
        "--use_transformer",
        action="store_true",
        help="Use Transformer for matching"
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.1,
        help="Target overlap ratio (default: 0.1 = 10%%)"
    )
    
    args = parser.parse_args()
    
    # SVG 파일 다운로드
    if args.download:
        print("Downloading SVG files from GitHub...")
        download_svgs()
    
    # 입력 디렉토리 확인
    input_path = Path(args.input_dir)
    if not input_path.exists():
        print(f"Error: Input directory {input_path} does not exist!")
        print("Use --download to download SVG files first.")
        return
    
    # SVG 파일 목록
    svg_files = sorted(input_path.glob("*.svg"))
    
    if not svg_files:
        print(f"No SVG files found in {input_path}")
        return
    
    if args.max_images:
        svg_files = svg_files[:args.max_images]
    
    print(f"Found {len(svg_files)} SVG files")
    
    # 스티처 초기화
    stitcher = SVGVectorStitcher(
        use_transformer=args.use_transformer,
        use_gnn=args.use_gnn,
        use_overlap_detection=True,
        layout_mode='auto'
    )
    
    # 스티칭 실행
    print(f"\nStarting panorama stitching...")
    print(f"Using {'GNN' if args.use_gnn else 'Transformer' if args.use_transformer else 'Basic'} matching")
    print(f"Target overlap: {args.overlap * 100:.1f}%")
    
    success = stitcher.create_panorama_svg(
        svg_files=[str(f) for f in svg_files],
        output_path=args.output,
        max_images=args.max_images
    )
    
    if success:
        print(f"\n✅ Success! Panorama SVG created: {args.output}")
    else:
        print(f"\n❌ Failed to create panorama SVG")


if __name__ == "__main__":
    main()

