"""
파노라마 스티칭 사용 예제
"""
from panorama_stitcher import PanoramaStitcher
import cv2
from pathlib import Path


def example_basic():
    """기본 사용 예제"""
    print("=== 기본 사용 예제 ===")
    
    # LoFTR 트랜스포머 모델 사용
    stitcher = PanoramaStitcher(
        feature_method='loftr',
        use_deep_learning=True,
        output_size=(2048, 1768)
    )
    
    # SVG 디렉토리에서 파노라마 생성
    panorama = stitcher.stitch_from_svg_dir(
        svg_dir='m2/',
        output_path='output_panorama_loftr.png',
        max_images=10  # 테스트용으로 10개만 사용
    )
    
    print(f"파노라마 크기: {panorama.shape}")


def example_traditional():
    """전통적인 방법 사용 예제 (딥러닝 없이)"""
    print("\n=== 전통적인 방법 예제 ===")
    
    # SIFT 사용
    stitcher = PanoramaStitcher(
        feature_method='sift',
        use_deep_learning=False,
        output_size=(2048, 1768)
    )
    
    panorama = stitcher.stitch_from_svg_dir(
        svg_dir='m2/',
        output_path='output_panorama_sift.png',
        max_images=10
    )
    
    print(f"파노라마 크기: {panorama.shape}")


def example_custom_images():
    """이미 변환된 이미지로 스티칭"""
    print("\n=== 커스텀 이미지 예제 ===")
    
    stitcher = PanoramaStitcher(
        feature_method='loftr',
        use_deep_learning=True
    )
    
    # 이미지 로드
    converter = stitcher.converter
    images = converter.convert_directory('m2/', max_images=5)
    
    # 순차 스티칭
    panorama = stitcher.stitch_sequence(images)
    
    # 저장
    cv2.imwrite('output_panorama_custom.png', 
                cv2.cvtColor(panorama, cv2.COLOR_RGB2BGR))
    
    print(f"파노라마 크기: {panorama.shape}")


if __name__ == '__main__':
    # 예제 실행
    try:
        example_basic()
    except Exception as e:
        print(f"기본 예제 오류: {e}")
    
    try:
        example_traditional()
    except Exception as e:
        print(f"전통적 방법 예제 오류: {e}")
    
    try:
        example_custom_images()
    except Exception as e:
        print(f"커스텀 이미지 예제 오류: {e}")



