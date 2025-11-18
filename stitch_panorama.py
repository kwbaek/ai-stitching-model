"""
파노라마 스티칭 실행 스크립트
"""
import argparse
import sys
from pathlib import Path
from panorama_stitcher import PanoramaStitcher


def main():
    parser = argparse.ArgumentParser(
        description='SVG 이미지를 딥러닝 기반으로 파노라마로 스티칭'
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
        default='output_panorama.png',
        help='출력 파일 경로'
    )
    parser.add_argument(
        '--method',
        type=str,
        default='loftr',
        choices=['loftr', 'disk', 'sift', 'orb'],
        help='특징점 매칭 방법'
    )
    parser.add_argument(
        '--use_deep_learning',
        action='store_true',
        default=True,
        help='딥러닝 모델 사용 (loftr, disk)'
    )
    parser.add_argument(
        '--no_deep_learning',
        dest='use_deep_learning',
        action='store_false',
        help='전통적인 방법 사용 (sift, orb)'
    )
    parser.add_argument(
        '--max_images',
        type=int,
        default=None,
        help='최대 이미지 수 (테스트용)'
    )
    parser.add_argument(
        '--output_size',
        type=int,
        nargs=2,
        default=[2048, 1768],
        help='SVG 변환 크기 (width height)'
    )
    
    args = parser.parse_args()
    
    # 입력 디렉토리 확인
    input_path = Path(args.input_dir)
    if not input_path.exists():
        print(f"Error: Input directory {args.input_dir} does not exist")
        sys.exit(1)
    
    # 스티처 생성
    stitcher = PanoramaStitcher(
        feature_method=args.method,
        use_deep_learning=args.use_deep_learning,
        output_size=tuple(args.output_size)
    )
    
    try:
        # 파노라마 생성
        panorama = stitcher.stitch_from_svg_dir(
            svg_dir=str(input_path),
            output_path=args.output,
            max_images=args.max_images
        )
        
        print(f"\nSuccess! Panorama created: {args.output}")
        print(f"Panorama size: {panorama.shape[1]}x{panorama.shape[0]}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()



