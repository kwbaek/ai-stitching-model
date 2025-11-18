"""
SVG 벡터 데이터 직접 스티칭 실행 스크립트
"""
import argparse
import sys
from pathlib import Path
from svg_vector_stitcher import SVGVectorStitcher


def main():
    parser = argparse.ArgumentParser(
        description='SVG 벡터 데이터를 직접 분석하여 파노라마로 스티칭'
    )
    parser.add_argument(
        '--input_dir',
        type=str,
        default='m2/',
        help='SVG 파일이 있는 디렉토리'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='output_panorama_vector.svg',
        help='출력 SVG 파일 경로'
    )
    parser.add_argument(
        '--max_images',
        type=int,
        default=None,
        help='최대 이미지 수 (테스트용)'
    )
    
    args = parser.parse_args()
    
    # 입력 디렉토리 확인
    input_path = Path(args.input_dir)
    if not input_path.exists():
        print(f"Error: Input directory {args.input_dir} does not exist")
        sys.exit(1)
    
    # SVG 파일 목록
    svg_files = sorted(input_path.glob('*.svg'))
    
    if len(svg_files) == 0:
        print(f"Error: No SVG files found in {args.input_dir}")
        sys.exit(1)
    
    if args.max_images:
        svg_files = svg_files[:args.max_images]
    
    print(f"Found {len(svg_files)} SVG files")
    
    # 스티처 생성
    stitcher = SVGVectorStitcher()
    
    try:
        # 파노라마 SVG 생성
        success = stitcher.create_panorama_svg(
            svg_files=[str(f) for f in svg_files],
            output_path=args.output,
            max_images=None  # 이미 필터링됨
        )
        
        if success:
            print(f"\nSuccess! Panorama SVG created: {args.output}")
        else:
            print("Error: Failed to create panorama SVG")
            sys.exit(1)
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

