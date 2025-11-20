"""
GitHub 저장소에서 SVG 파일 다운로드 및 처리
"""
import os
import subprocess
import shutil
from pathlib import Path
from typing import List, Optional
import argparse


def clone_repository(repo_url: str, target_dir: str = "github_repo") -> Path:
    """
    GitHub 저장소 클론
    
    Args:
        repo_url: 저장소 URL
        target_dir: 저장소를 클론할 디렉토리
        
    Returns:
        클론된 저장소 경로
    """
    target_path = Path(target_dir)
    
    # 이미 존재하면 삭제
    if target_path.exists():
        print(f"Removing existing directory: {target_path}")
        shutil.rmtree(target_path)
    
    print(f"Cloning repository: {repo_url}")
    try:
        subprocess.run(
            ["git", "clone", repo_url, str(target_path)],
            check=True,
            capture_output=True,
            text=True
        )
        print(f"Successfully cloned to {target_path}")
        return target_path
    except subprocess.CalledProcessError as e:
        print(f"Error cloning repository: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        raise


def find_svg_files(directory: Path, recursive: bool = True) -> List[Path]:
    """
    디렉토리에서 SVG 파일 찾기
    
    Args:
        directory: 검색할 디렉토리
        recursive: 재귀적으로 검색할지 여부
        
    Returns:
        SVG 파일 경로 리스트
    """
    if recursive:
        svg_files = list(directory.rglob("*.svg"))
    else:
        svg_files = list(directory.glob("*.svg"))
    
    # 정렬
    svg_files.sort()
    
    return svg_files


def copy_svg_files(svg_files: List[Path], output_dir: str = "input_svgs") -> List[str]:
    """
    SVG 파일들을 출력 디렉토리로 복사
    
    Args:
        svg_files: 원본 SVG 파일 경로 리스트
        output_dir: 출력 디렉토리
        
    Returns:
        복사된 파일 경로 리스트
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    copied_files = []
    
    for i, svg_file in enumerate(svg_files, 1):
        # 파일명 생성 (순서대로)
        new_name = f"svg_{i:04d}.svg"
        dest_path = output_path / new_name
        
        # 복사
        shutil.copy2(svg_file, dest_path)
        copied_files.append(str(dest_path))
        
        if i % 10 == 0:
            print(f"Copied {i}/{len(svg_files)} files...")
    
    print(f"Total {len(copied_files)} SVG files copied to {output_path}")
    return copied_files


def main():
    parser = argparse.ArgumentParser(
        description="Download and process SVG files from GitHub repository"
    )
    parser.add_argument(
        "--repo",
        type=str,
        default="https://github.com/emsec/unsupervised-ic-sem-segmentation-extended",
        help="GitHub repository URL"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="input_svgs",
        help="Output directory for SVG files"
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="Maximum number of SVG files to process"
    )
    parser.add_argument(
        "--clone_dir",
        type=str,
        default="github_repo",
        help="Directory to clone repository"
    )
    
    args = parser.parse_args()
    
    # 저장소 클론
    repo_path = clone_repository(args.repo, args.clone_dir)
    
    # SVG 파일 찾기
    print("Searching for SVG files...")
    svg_files = find_svg_files(repo_path, recursive=True)
    
    if not svg_files:
        print("No SVG files found in repository!")
        return
    
    print(f"Found {len(svg_files)} SVG files")
    
    # 최대 파일 수 제한
    if args.max_files:
        svg_files = svg_files[:args.max_files]
        print(f"Limited to {len(svg_files)} files")
    
    # 파일 복사
    copied_files = copy_svg_files(svg_files, args.output)
    
    print(f"\n✅ Successfully processed {len(copied_files)} SVG files")
    print(f"Output directory: {args.output}")
    
    return copied_files


if __name__ == "__main__":
    main()

