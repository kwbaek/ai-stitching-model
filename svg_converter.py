"""
SVG 벡터 이미지를 래스터 이미지로 변환하는 모듈
"""
import os
import io
import numpy as np
from PIL import Image
import cairosvg
from pathlib import Path
from typing import List, Tuple
import cv2


class SVGConverter:
    """SVG 파일을 래스터 이미지로 변환하는 클래스"""
    
    def __init__(self, output_size: Tuple[int, int] = (2048, 1768), dpi: int = 300):
        """
        Args:
            output_size: 출력 이미지 크기 (width, height)
            dpi: 변환 해상도
        """
        self.output_size = output_size
        self.dpi = dpi
    
    def svg_to_image(self, svg_path: str) -> np.ndarray:
        """
        SVG 파일을 numpy 배열로 변환
        
        Args:
            svg_path: SVG 파일 경로
            
        Returns:
            RGB 이미지 배열 (H, W, 3)
        """
        # SVG를 PNG로 변환
        png_data = cairosvg.svg2png(
            url=svg_path,
            output_width=self.output_size[0],
            output_height=self.output_size[1],
            dpi=self.dpi
        )
        
        # PNG 데이터를 PIL Image로 변환
        image = Image.open(io.BytesIO(png_data))
        
        # RGB로 변환 (RGBA인 경우)
        if image.mode == 'RGBA':
            # 흰색 배경에 합성
            background = Image.new('RGB', image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3])
            image = background
        elif image.mode != 'RGB':
            image = image.convert('RGB')
        
        # numpy 배열로 변환
        return np.array(image)
    
    def convert_directory(self, input_dir: str, output_dir: str = None, 
                         max_images: int = None) -> List[np.ndarray]:
        """
        디렉토리 내 모든 SVG 파일을 변환
        
        Args:
            input_dir: 입력 디렉토리
            output_dir: 출력 디렉토리 (None이면 메모리에만 저장)
            max_images: 최대 변환할 이미지 수 (None이면 모두 변환)
            
        Returns:
            변환된 이미지 리스트
        """
        input_path = Path(input_dir)
        svg_files = sorted(input_path.glob('*.svg'))
        
        if max_images:
            svg_files = svg_files[:max_images]
        
        images = []
        for svg_file in svg_files:
            try:
                image = self.svg_to_image(str(svg_file))
                images.append(image)
                
                if output_dir:
                    output_path = Path(output_dir)
                    output_path.mkdir(parents=True, exist_ok=True)
                    output_file = output_path / f"{svg_file.stem}.png"
                    Image.fromarray(image).save(output_file)
                    
            except Exception as e:
                print(f"Error converting {svg_file}: {e}")
                continue
        
        return images

