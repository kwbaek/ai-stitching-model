#!/usr/bin/env python3
"""
래스터 기반 딥러닝 모델별 스티칭 테스트 및 결과 비교
"""
import sys
import time
from pathlib import Path
from svg_vector_stitcher import SVGVectorStitcher

def test_raster_model(method, max_images=10):
    """특정 래스터 모델로 스티칭 테스트"""
    model_names = {
        'loftr': 'LoFTR',
        'disk': 'DISK',
        'lightglue': 'LightGlue',
        'lightglue_disk': 'LightGlue+DISK',
        'dinov2': 'DINOv2'
    }
    
    model_name = model_names.get(method, method)
    
    print(f"\n{'='*60}")
    print(f"Testing: {model_name} ({method})")
    print(f"{'='*60}")
    
    input_dir = 'm2/'
    output_file = f'panorama_raster_{method}.svg'
    
    # SVG 파일 목록
    svg_files = sorted(Path(input_dir).glob('*.svg'))
    if len(svg_files) == 0:
        print(f"Error: No SVG files found in {input_dir}")
        return None
    
    if max_images:
        svg_files = svg_files[:max_images]
    
    print(f"Found {len(svg_files)} SVG files")
    
    # 스티처 생성 (래스터 기반 딥러닝 매칭 사용)
    try:
        stitcher = SVGVectorStitcher(
            use_raster_matching=True,
            raster_method=method,
            use_overlap_detection=True
        )
    except Exception as e:
        print(f"❌ Failed to initialize {method}: {e}")
        return {
            'model': model_name,
            'method': method,
            'success': False,
            'time': 0,
            'file_size': 0,
            'output': None,
            'error': str(e)
        }
    
    # 시간 측정
    start_time = time.time()
    
    try:
        # 파노라마 SVG 생성
        success = stitcher.create_panorama_svg(
            svg_files=[str(f) for f in svg_files],
            output_path=output_file,
            max_images=None
        )
        
        elapsed_time = time.time() - start_time
        
        if success:
            # 파일 크기 확인
            file_size = Path(output_file).stat().st_size / (1024 * 1024)  # MB
            print(f"\n✅ Success!")
            print(f"   Output: {output_file}")
            print(f"   Time: {elapsed_time:.2f} seconds")
            print(f"   File size: {file_size:.2f} MB")
            return {
                'model': model_name,
                'method': method,
                'success': True,
                'time': elapsed_time,
                'file_size': file_size,
                'output': output_file
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
    print("\n💡 SVG → 래스터 변환 → 딥러닝 매칭 → 스티칭")
    print("   벡터 기반보다 더 정확한 매칭을 제공합니다.")
    
    max_images = 10  # 테스트용 이미지 수
    
    results = []
    
    # 사용 가능한 모델들 테스트
    methods_to_test = ['loftr', 'disk']
    
    # LightGlue 테스트 (설치되어 있으면)
    try:
        from lightglue import LightGlue
        methods_to_test.extend(['lightglue', 'lightglue_disk'])
        print("\n✅ LightGlue available")
    except:
        print("\n⚠️  LightGlue not available (skip)")
    
    # DINOv2 테스트 (사용 가능하면)
    try:
        from transformers import AutoImageProcessor, AutoModel
        methods_to_test.append('dinov2')
        print("✅ DINOv2 available")
    except:
        print("⚠️  DINOv2 not available (skip)")
    
    for method in methods_to_test:
        result = test_raster_model(method, max_images)
        if result:
            results.append(result)
    
    # 결과 요약
    print(f"\n{'='*60}")
    print("결과 요약")
    print(f"{'='*60}")
    print(f"{'Model':<25} {'Status':<10} {'Time (s)':<12} {'Size (MB)':<12}")
    print("-" * 60)
    
    for r in results:
        if r:
            status = "✅ Success" if r['success'] else "❌ Failed"
            time_str = f"{r['time']:.2f}" if r['time'] > 0 else "N/A"
            size_str = f"{r['file_size']:.2f}" if r['file_size'] > 0 else "N/A"
            print(f"{r['model']:<25} {status:<10} {time_str:<12} {size_str:<12}")
    
    print(f"\n{'='*60}")
    print("생성된 파일:")
    print(f"{'='*60}")
    for r in results:
        if r and r['success'] and r['output']:
            print(f"  - {r['output']}")
    
    print(f"\n✅ 테스트 완료!")
    print("\n💡 추천: 래스터 기반 딥러닝 모델이 벡터 기반보다 더 정확한 매칭을 제공합니다.")

if __name__ == '__main__':
    main()

