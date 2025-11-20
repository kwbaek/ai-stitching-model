"""
딥러닝 기반 특징점 추출 및 매칭 모듈
LoFTR, DISK, LightGlue, DINOv2 등 지원
"""
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
import kornia as K
from kornia.feature import LoFTR, DISK
from kornia.utils import image_to_tensor

# Optional: LightGlue (더 빠르고 정확한 매칭)
try:
    from lightglue import LightGlue, SuperPoint, DISK as LightGlueDISK
    from lightglue.utils import load_image, rbd
    LIGHTGLUE_AVAILABLE = True
except ImportError:
    LIGHTGLUE_AVAILABLE = False
    print("LightGlue not available. Install with: pip install lightglue")

# Optional: DINOv2 (강력한 특징 추출)
try:
    from transformers import AutoImageProcessor, AutoModel
    DINOV2_AVAILABLE = True
except ImportError:
    DINOV2_AVAILABLE = False
    print("DINOv2 not available. Install with: pip install transformers")


class DeepFeatureMatcher:
    """딥러닝 기반 특징점 매칭 클래스"""
    
    def __init__(self, method: str = 'loftr', device: str = None):
        """
        Args:
            method: 'loftr', 'disk', 'lightglue', 'lightglue_disk', 'dinov2'
            device: 'cuda' 또는 'cpu'
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.method = method
        
        if method == 'loftr':
            self.model = LoFTR(pretrained='outdoor').to(self.device).eval()
            self.extractor = None
        elif method == 'disk':
            self.model = DISK.from_pretrained('depth').to(self.device).eval()
            self.extractor = None
        elif method == 'lightglue' and LIGHTGLUE_AVAILABLE:
            self.extractor = SuperPoint(max_num_keypoints=2048).eval().to(self.device)
            self.model = LightGlue(features='superpoint').eval().to(self.device)
        elif method == 'lightglue_disk' and LIGHTGLUE_AVAILABLE:
            self.extractor = LightGlueDISK(max_num_keypoints=2048).eval().to(self.device)
            self.model = LightGlue(features='disk').eval().to(self.device)
        elif method == 'dinov2' and DINOV2_AVAILABLE:
            self.processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
            self.model = AutoModel.from_pretrained('facebook/dinov2-base').to(self.device).eval()
            self.extractor = None
        else:
            available_methods = ['loftr', 'disk']
            if LIGHTGLUE_AVAILABLE:
                available_methods.extend(['lightglue', 'lightglue_disk'])
            if DINOV2_AVAILABLE:
                available_methods.append('dinov2')
            raise ValueError(f"Unknown method: {method}. Available: {available_methods}")
    
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
        with torch.no_grad():
            if self.method in ['lightglue', 'lightglue_disk'] and LIGHTGLUE_AVAILABLE:
                # LightGlue 매칭 (더 빠르고 정확함)
                # 이미지를 PIL Image로 변환
                from PIL import Image
                if isinstance(img1, np.ndarray):
                    img1_pil = Image.fromarray(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB) if len(img1.shape) == 3 else img1)
                    img2_pil = Image.fromarray(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB) if len(img2.shape) == 3 else img2)
                else:
                    img1_pil, img2_pil = img1, img2
                
                # 이미지 로드 및 전처리
                image0 = load_image(img1_pil).to(self.device)
                image1 = load_image(img2_pil).to(self.device)
                
                # 특징점 추출
                feats0 = self.extractor.extract(image0)
                feats1 = self.extractor.extract(image1)
                
                # 매칭
                matches01 = self.model({'image0': feats0, 'image1': feats1})
                
                # 매칭점 추출
                mkpts0 = feats0['keypoints'][matches01['matches'][:, 0]].cpu().numpy()
                mkpts1 = feats1['keypoints'][matches01['matches'][:, 1]].cpu().numpy()
                mconf = matches01['match_confidence'].cpu().numpy()
                
            elif self.method == 'dinov2' and DINOV2_AVAILABLE:
                # DINOv2 특징 추출 및 매칭
                from PIL import Image
                if isinstance(img1, np.ndarray):
                    img1_pil = Image.fromarray(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB) if len(img1.shape) == 3 else img1)
                    img2_pil = Image.fromarray(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB) if len(img2.shape) == 3 else img2)
                else:
                    img1_pil, img2_pil = img1, img2
                
                # 특징 추출
                inputs1 = self.processor(images=img1_pil, return_tensors="pt").to(self.device)
                inputs2 = self.processor(images=img2_pil, return_tensors="pt").to(self.device)
                
                outputs1 = self.model(**inputs1)
                outputs2 = self.model(**inputs2)
                
                # 패치 특징 추출 (평균 풀링)
                feat1 = outputs1.last_hidden_state.mean(dim=1).cpu().numpy()
                feat2 = outputs2.last_hidden_state.mean(dim=1).cpu().numpy()
                
                # 간단한 매칭 (코사인 유사도)
                from scipy.spatial.distance import cdist
                distances = cdist(feat1, feat2, metric='cosine')
                matches = np.argmin(distances, axis=1)
                
                # 그리드 좌표 생성 (DINOv2는 패치 기반)
                h1, w1 = img1.shape[:2]
                h2, w2 = img2.shape[:2]
                patch_size = 14  # DINOv2 base patch size
                
                # 간단한 그리드 매칭 (실제로는 더 정교한 방법 필요)
                mkpts0 = []
                mkpts1 = []
                mconf = []
                
                for i, j in enumerate(matches):
                    if distances[i, j] < 0.5:  # 유사도 임계값
                        # 패치 위치를 픽셀 좌표로 변환
                        patch_h1 = i // (w1 // patch_size)
                        patch_w1 = i % (w1 // patch_size)
                        patch_h2 = j // (w2 // patch_size)
                        patch_w2 = j % (w2 // patch_size)
                        
                        mkpts0.append([patch_w1 * patch_size, patch_h1 * patch_size])
                        mkpts1.append([patch_w2 * patch_size, patch_h2 * patch_size])
                        mconf.append(1.0 - distances[i, j])
                
                mkpts0 = np.array(mkpts0) if mkpts0 else np.array([])
                mkpts1 = np.array(mkpts1) if mkpts1 else np.array([])
                mconf = np.array(mconf) if mconf else np.array([])
                
            else:
                # 기존 LoFTR/DISK 방식
                # 이미지 전처리 (이미 (1, H, W) 형태)
                img1_tensor = self.preprocess_image(img1)  # (1, H, W)
                img2_tensor = self.preprocess_image(img2)  # (1, H, W)
                
                # 배치 차원 추가하여 (B, 1, H, W) 형태로
                img1_batch = img1_tensor.unsqueeze(0)  # (1, 1, H, W)
                img2_batch = img2_tensor.unsqueeze(0)  # (1, 1, H, W)
                
                if self.method == 'loftr':
                    # LoFTR 매칭
                    input_dict = {
                        'image0': img1_batch,
                        'image1': img2_batch
                    }
                    correspondences = self.model(input_dict)
                    
                    # 매칭점 추출
                    if 'mkpts0_f' in correspondences:
                        mkpts0 = correspondences['mkpts0_f'].cpu().numpy()
                        mkpts1 = correspondences['mkpts1_f'].cpu().numpy()
                        mconf = correspondences['mconf'].cpu().numpy()
                    elif 'keypoints0' in correspondences:
                        mkpts0 = correspondences['keypoints0'].cpu().numpy()
                        mkpts1 = correspondences['keypoints1'].cpu().numpy()
                        mconf = correspondences['confidence'].cpu().numpy()
                    else:
                        print(f"Warning: LoFTR output keys: {correspondences.keys()}")
                        mkpts0 = np.array([])
                        mkpts1 = np.array([])
                        mconf = np.array([])
                    
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

