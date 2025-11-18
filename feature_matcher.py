"""
딥러닝 기반 특징점 추출 및 매칭 모듈
SuperPoint + SuperGlue 또는 LoFTR 사용
"""
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
import kornia as K
from kornia.feature import LoFTR, DISK
from kornia.utils import image_to_tensor


class DeepFeatureMatcher:
    """딥러닝 기반 특징점 매칭 클래스"""
    
    def __init__(self, method: str = 'loftr', device: str = None):
        """
        Args:
            method: 'loftr' 또는 'superpoint_superglue'
            device: 'cuda' 또는 'cpu'
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.method = method
        
        if method == 'loftr':
            self.model = LoFTR(pretrained='outdoor').to(self.device).eval()
        elif method == 'disk':
            self.model = DISK.from_pretrained('depth').to(self.device).eval()
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """이미지를 모델 입력 형식으로 전처리"""
        # RGB to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Normalize and convert to tensor
        # image_to_tensor는 grayscale을 (1, H, W) 형태로 변환
        img_tensor = image_to_tensor(gray, False).float() / 255.0
        img_tensor = img_tensor.to(self.device)
        
        # image_to_tensor가 (1, H, W) 또는 (1, 1, H, W) 형태로 반환할 수 있음
        # LoFTR는 (1, H, W) 형태를 기대하므로 정규화
        if len(img_tensor.shape) == 4:
            # (1, 1, H, W) -> (1, H, W)
            img_tensor = img_tensor.squeeze(0)
        elif len(img_tensor.shape) == 2:
            # (H, W) -> (1, H, W)
            img_tensor = img_tensor.unsqueeze(0)
        elif len(img_tensor.shape) == 3 and img_tensor.shape[0] != 1:
            # (C, H, W) where C != 1 -> (1, H, W)
            if img_tensor.shape[0] == 3:
                img_tensor = img_tensor.mean(dim=0, keepdim=True)
            else:
                img_tensor = img_tensor[:1]  # 첫 번째 채널만 사용
        
        return img_tensor
    
    def match_features(self, img1: np.ndarray, img2: np.ndarray) -> Dict:
        """
        두 이미지 간 특징점 매칭
        
        Args:
            img1: 첫 번째 이미지 (H, W, 3)
            img2: 두 번째 이미지 (H, W, 3)
            
        Returns:
            매칭 결과 딕셔너리
        """
        # 이미지 전처리 (이미 (1, H, W) 형태)
        img1_tensor = self.preprocess_image(img1)  # (1, H, W)
        img2_tensor = self.preprocess_image(img2)  # (1, H, W)
        
        # 배치 차원 추가하여 (B, 1, H, W) 형태로
        img1_batch = img1_tensor.unsqueeze(0)  # (1, 1, H, W)
        img2_batch = img2_tensor.unsqueeze(0)  # (1, 1, H, W)
        
        with torch.no_grad():
            if self.method == 'loftr':
                # LoFTR 매칭
                input_dict = {
                    'image0': img1_batch,
                    'image1': img2_batch
                }
                correspondences = self.model(input_dict)
                
                # 매칭점 추출
                mkpts0 = correspondences['mkpts0_f'].cpu().numpy()
                mkpts1 = correspondences['mkpts1_f'].cpu().numpy()
                mconf = correspondences['mconf'].cpu().numpy()
                
            elif self.method == 'disk':
                # DISK 특징점 추출 및 매칭
                features1 = self.model(img1_batch)
                features2 = self.model(img2_batch)
                
                # 특징점 매칭
                matches = self.model.match(features1, features2)
                
                mkpts0 = matches['keypoints0'].cpu().numpy()
                mkpts1 = matches['keypoints1'].cpu().numpy()
                mconf = matches['confidence'].cpu().numpy()
        
        return {
            'keypoints0': mkpts0,
            'keypoints1': mkpts1,
            'confidence': mconf,
            'num_matches': len(mkpts0)
        }


class TraditionalFeatureMatcher:
    """전통적인 특징점 매칭 (SIFT/ORB) - 대안"""
    
    def __init__(self, method: str = 'sift', max_features: int = 5000):
        """
        Args:
            method: 'sift' 또는 'orb'
            max_features: 최대 특징점 수
        """
        self.method = method
        self.max_features = max_features
        
        if method == 'sift':
            self.detector = cv2.SIFT_create(nfeatures=max_features)
            self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        elif method == 'orb':
            self.detector = cv2.ORB_create(nfeatures=max_features)
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def match_features(self, img1: np.ndarray, img2: np.ndarray) -> Dict:
        """두 이미지 간 특징점 매칭"""
        # 그레이스케일 변환
        if len(img1.shape) == 3:
            gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
        else:
            gray1, gray2 = img1, img2
        
        # 특징점 추출
        kp1, des1 = self.detector.detectAndCompute(gray1, None)
        kp2, des2 = self.detector.detectAndCompute(gray2, None)
        
        if des1 is None or des2 is None or len(des1) < 4 or len(des2) < 4:
            return {
                'keypoints0': np.array([]),
                'keypoints1': np.array([]),
                'confidence': np.array([]),
                'num_matches': 0
            }
        
        # 매칭
        matches = self.matcher.knnMatch(des1, des2, k=2)
        
        # Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
        
        # 매칭점 추출
        if len(good_matches) < 4:
            return {
                'keypoints0': np.array([]),
                'keypoints1': np.array([]),
                'confidence': np.array([]),
                'num_matches': 0
            }
        
        pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])
        conf = np.array([1.0 / (1.0 + m.distance) for m in good_matches])
        
        return {
            'keypoints0': pts1,
            'keypoints1': pts2,
            'confidence': conf,
            'num_matches': len(good_matches)
        }

