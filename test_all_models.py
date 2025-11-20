#!/usr/bin/env python3
"""
모든 모델별로 스티칭 테스트 및 결과 비교
"""
import sys
import time
from pathlib import Path
from svg_vector_stitcher import SVGVectorStitcher

def test_model(model_name, use_transformer=False, use_gnn=False, max_images=10):
    """특정 모델로 스티칭 테스트"""
    print(f"\n{'='*60}")
    print(f"Testing: {model_name}")
    print(f"{'='*60}")
    
    input_dir = 'm2/'
    output_file = f'panorama_{model_name.lower().replace(" ", "_")}.svg'
    
    # SVG 파일 목록
    svg_files = sorted(Path(input_dir).glob('*.svg'))
    if len(svg_files) == 0:
        print(f"Error: No SVG files found in {input_dir}")
        return None
    
    if max_images:
        svg_files = svg_files[:max_images]
    
    print(f"Found {len(svg_files)} SVG files")
    
    # 스티처 생성
    stitcher = SVGVectorStitcher(
        use_transformer=use_transformer,
        use_gnn=use_gnn,
        use_overlap_detection=True
    )
    
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
                'success': True,
                'time': elapsed_time,
                'file_size': file_size,
                'output': output_file
            }
        else:
            print(f"\n❌ Failed")
            return {
                'model': model_name,
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
            'success': False,
            'time': elapsed_time,
            'file_size': 0,
            'output': None,
            'error': str(e)
        }

def main():
    print("="*60)
    print("모델별 스티칭 결과 비교 테스트")
    print("="*60)
    
    max_images = 10  # 테스트용 이미지 수
    
    results = []
    
    # 1. 기본 벡터 매칭
    results.append(test_model("Basic Vector Matching", False, False, max_images))
    
    # 2. Transformer 기반 매칭
    try:
        from transformer_vector_matcher import TransformerVectorMatcher
        results.append(test_model("Transformer Vector Matching", True, False, max_images))
    except Exception as e:
        print(f"\n⚠️  Transformer model not available: {e}")
    
    # 3. GNN 기반 매칭
    try:
        from graph_vector_matcher import GraphVectorMatcher
        results.append(test_model("GNN Vector Matching", False, True, max_images))
    except Exception as e:
        print(f"\n⚠️  GNN model not available: {e}")
    
    # 결과 요약
    print(f"\n{'='*60}")
    print("결과 요약")
    print(f"{'='*60}")
    print(f"{'Model':<30} {'Status':<10} {'Time (s)':<12} {'Size (MB)':<12}")
    print("-" * 60)
    
    for r in results:
        if r:
            status = "✅ Success" if r['success'] else "❌ Failed"
            time_str = f"{r['time']:.2f}" if r['time'] > 0 else "N/A"
            size_str = f"{r['file_size']:.2f}" if r['file_size'] > 0 else "N/A"
            print(f"{r['model']:<30} {status:<10} {time_str:<12} {size_str:<12}")
    
    print(f"\n{'='*60}")
    print("생성된 파일:")
    print(f"{'='*60}")
    for r in results:
        if r and r['success'] and r['output']:
            print(f"  - {r['output']}")
    
    print(f"\n✅ 테스트 완료!")

if __name__ == '__main__':
    main()

