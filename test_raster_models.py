#!/usr/bin/env python3
"""
래스터 기반 딥러닝 모델별 스티칭 테스트 및 결과 비교
SVG를 래스터로 변환 후 딥러닝 모델 사용
"""
import sys
import time
from pathlib import Path
from panorama_stitcher import PanoramaStitcher

def test_raster_model(model_name, method, max_images=10):
    """특정 래스터 모델로 스티칭 테스트"""
    print(f"\n{'='*60}")
    print(f"Testing: {model_name} ({method})")
    print(f"{'='*60}")
    
    input_dir = 'm2/'
    output_file = f'panorama_raster_{method}.png'
    
    # 스티처 생성
    stitcher = PanoramaStitcher(
        feature_method=method,
        use_deep_learning=True,
        output_size=(4096, 3536)  # 원본 크기 유지
    )
    
    # 시간 측정
    start_time = time.time()
    
    try:
        # 파노라마 생성
        panorama = stitcher.stitch_from_svg_dir(
            svg_dir=input_dir,
            output_path=output_file,
            max_images=max_images
        )
        
        elapsed_time = time.time() - start_time
        
        if panorama is not None:
            # 파일 크기 확인
            file_size = Path(output_file).stat().st_size / (1024 * 1024)  # MB
            h, w = panorama.shape[:2]
            print(f"\n✅ Success!")
            print(f"   Output: {output_file}")
            print(f"   Panorama size: {w} x {h}")
            print(f"   Time: {elapsed_time:.2f} seconds")
            print(f"   File size: {file_size:.2f} MB")
            return {
                'model': model_name,
                'method': method,
                'success': True,
                'time': elapsed_time,
                'file_size': file_size,
                'output': output_file,
                'size': (w, h)
            }
        else:
            print(f"\n❌ Failed: No panorama generated")
            return {
                'model': model_name,
                'method': method,
                'success': False,
                'time': elapsed_time,
                'file_size': 0,
                'output': None
            }
            
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return {
            'model': model_name,
            'method': method,
            'success': False,
            'time': elapsed_time,
            'file_size': 0,
            'output': None,
            'error': str(e)
        }

def main():
    print("="*60)
    print("래스터 기반 딥러닝 모델별 스티칭 결과 비교 테스트")
    print("="*60)
    print("\n⚠️  벡터 기반이 아닌 래스터 기반 딥러닝 모델 사용")
    print("   SVG → 래스터 변환 → 딥러닝 매칭 → 스티칭")
    
    max_images = 10  # 테스트용 이미지 수
    
    results = []
    
    # 1. LoFTR (Transformer 기반)
    print("\n" + "="*60)
    print("1. LoFTR 테스트")
    print("="*60)
    results.append(test_raster_model("LoFTR", "loftr", max_images))
    
    # 2. DISK
    print("\n" + "="*60)
    print("2. DISK 테스트")
    print("="*60)
    results.append(test_raster_model("DISK", "disk", max_images))
    
    # 3. SIFT (전통적인 방법, 비교용)
    print("\n" + "="*60)
    print("3. SIFT 테스트 (비교용)")
    print("="*60)
    results.append(test_raster_model("SIFT", "sift", max_images))
    
    # 결과 요약
    print(f"\n{'='*60}")
    print("결과 요약")
    print(f"{'='*60}")
    print(f"{'Model':<20} {'Status':<10} {'Time (s)':<12} {'Size (MB)':<12} {'Image Size':<20}")
    print("-" * 80)
    
    for r in results:
        if r:
            status = "✅ Success" if r['success'] else "❌ Failed"
            time_str = f"{r['time']:.2f}" if r['time'] > 0 else "N/A"
            size_str = f"{r['file_size']:.2f}" if r['file_size'] > 0 else "N/A"
            img_size = f"{r.get('size', (0,0))[0]}x{r.get('size', (0,0))[1]}" if r.get('size') else "N/A"
            print(f"{r['model']:<20} {status:<10} {time_str:<12} {size_str:<12} {img_size:<20}")
    
    print(f"\n{'='*60}")
    print("생성된 파일:")
    print(f"{'='*60}")
    for r in results:
        if r and r['success'] and r['output']:
            print(f"  - {r['output']}")
    
    print(f"\n✅ 테스트 완료!")
    print("\n💡 추천: 래스터 기반 딥러닝 모델이 벡터 기반보다 더 정확한 매칭을 제공할 수 있습니다.")

if __name__ == '__main__':
    main()

