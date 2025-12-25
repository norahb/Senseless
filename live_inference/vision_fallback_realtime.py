"""
Real-Time Vision Fallback Module

Captures images on-demand when sensor confidence is low or sensors error.
Runs vision models on captured images.
Optionally discards images after inference based on configuration.
"""

import os
import sys
import json
import numpy as np
import torch
import time
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple
from datetime import datetime
from PIL import Image
import cv2

# Add training path for model imports
training_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'training'))
if training_path not in sys.path:
    sys.path.insert(0, training_path)

logger = logging.getLogger(__name__)


class RealtimeVisionFallback:
    """
    Real-time vision fallback handler.
    
    Features:
    - Capture images on-demand (low confidence or sensor error)
    - Run vision models for anomaly detection
    - Optional image persistence (configurable)
    - Save only anomaly images if configured
    - Model loaded once at startup
    """
    
    def __init__(self, config, camera_source: int = 0):
        """
        Initialize real-time vision fallback.
        
        Args:
            config: LiveInferenceConfig instance
            camera_source: Camera index (0 for default webcam)
        """
        self.config = config
        self.use_case = config.use_case
        self.camera_source = camera_source
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.image_transform = None
        self.inference_count = 0
        self.successful_captures = 0
        self.failed_captures = 0
        self.inference_times = []
        self.last_opened_index = None
        
        # Setup image preprocessing
        self._setup_transforms()
        
        # Load vision model at startup
        self._load_vision_model()
    
    def _setup_transforms(self):
        """Setup image preprocessing transforms."""
        from torchvision import transforms
        
        # Standard ImageNet normalization
        MEAN = [0.485, 0.456, 0.406]
        STD = [0.229, 0.224, 0.225]
        
        self.image_transform = transforms.Compose([
            transforms.Resize(self.config.image_capture_resolution),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])
        
        logger.info(f"📸 Image transforms setup: {self.config.image_capture_resolution}")
    
    def _load_vision_model(self):
        """Load vision model once at startup."""
        try:
            vision_model_path = Path(self.config.vision_model_path)
            
            if not vision_model_path.exists():
                logger.warning(f"⚠️ Vision model not found: {vision_model_path}")
                return
            
            logger.info(f"🧠 Loading vision model for {self.use_case}...")
            
            # Load based on file extension
            if str(vision_model_path).endswith('.pth'):
                # PyTorch model (MobileNetV2)
                self._load_pytorch_model(str(vision_model_path))
            
            elif str(vision_model_path).endswith('.pt'):
                # TorchScript or different PyTorch format (EfficientNet)
                self._load_torchscript_model(str(vision_model_path))
            
            else:
                logger.warning(f"⚠️ Unknown model format: {vision_model_path}")
                self.model = None
        
        except Exception as e:
            logger.error(f"❌ Failed to load vision model: {e}")
            self.model = None
    
    def _load_pytorch_model(self, model_path: str):
        """Load standard PyTorch model (MobileNetV2)."""
        try:
            from torchvision import models
            
            # Create MobileNetV2
            self.model = models.mobilenet_v2(pretrained=False, num_classes=2)
            
            # Load weights
            checkpoint = torch.load(model_path, map_location=self.device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            
            self.model.eval().to(self.device)
            logger.info(f"✅ Loaded PyTorch model from {model_path}")
        
        except Exception as e:
            logger.error(f"❌ Failed to load PyTorch model: {e}")
            self.model = None
    
    def _load_torchscript_model(self, model_path: str):
        """Load TorchScript or EfficientNet model."""
        try:
            # Try TorchScript first
            try:
                self.model = torch.jit.load(model_path, map_location=self.device)
                self.model.eval()
                logger.info(f"✅ Loaded TorchScript model from {model_path}")
            except:
                # Try loading as standard PyTorch
                checkpoint = torch.load(model_path, map_location=self.device)
                if isinstance(checkpoint, dict):
                    # Assume it's a state dict
                    from torchvision import models
                    self.model = models.mobilenet_v2(pretrained=False, num_classes=2)
                    self.model.load_state_dict(checkpoint)
                else:
                    self.model = checkpoint
                
                self.model.eval().to(self.device)
                logger.info(f"✅ Loaded model from {model_path}")
        
        except Exception as e:
            logger.error(f"❌ Failed to load TorchScript model: {e}")
            self.model = None
    
    def capture_image(self) -> Optional[Tuple[np.ndarray, str]]:
        """
        Capture single image from camera.
        
        Returns:
            Tuple of (image_array, timestamp_str) or None if capture fails
        """
        try:
            # Try primary source, then fallbacks
            indices_to_try = [self.camera_source] + [i for i in self.config.camera_fallback_indices if i != self.camera_source]
            cap = None
            opened_index = None
            for idx in indices_to_try:
                temp_cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW) if os.name == 'nt' else cv2.VideoCapture(idx)
                if temp_cap is not None and temp_cap.isOpened():
                    cap = temp_cap
                    opened_index = idx
                    self.last_opened_index = opened_index
                    break
                if temp_cap is not None:
                    temp_cap.release()

            if cap is None or not cap.isOpened():
                logger.warning(f"⚠️ Cannot open camera {self.camera_source} (tried {indices_to_try})")
                self.failed_captures += 1
                return None
            
            if not cap.isOpened():
                logger.warning(f"⚠️ Cannot open camera {self.camera_source}")
                self.failed_captures += 1
                return None
            
            # Set timeout
            start_time = time.time()
            timeout = self.config.image_capture_timeout
            
            ret = False
            frame = None
            while (time.time() - start_time) < timeout:
                ret, frame = cap.read()
                if ret:
                    break
                time.sleep(0.05)
            
            cap.release()
            
            if not ret or frame is None:
                logger.warning(f"⚠️ Failed to capture frame from camera {self.camera_source}")
                self.failed_captures += 1
                return None
            
            # Resize to target resolution
            target_h, target_w = self.config.image_capture_resolution
            frame = cv2.resize(frame, (target_w, target_h))
            
            # Optional preview window
            if getattr(self.config, 'preview_capture', False):
                try:
                    preview_bgr = frame.copy()
                    cv2.imshow('Live Preview', preview_bgr)
                    delay = int(getattr(self.config, 'preview_delay_ms', 500))
                    cv2.waitKey(delay)
                    cv2.destroyWindow('Live Preview')
                except Exception:
                    pass

            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Generate timestamp
            timestamp = datetime.now().isoformat()
            
            self.successful_captures += 1
            logger.debug(f"📸 Captured image: {timestamp}")
            
            return (frame, timestamp)
        
        except Exception as e:
            logger.error(f"❌ Image capture failed: {e}")
            self.failed_captures += 1
            return None
    
    def infer_image(self, image_array: np.ndarray) -> Dict:
        """
        Run vision inference on image.
        
        Args:
            image_array: Image as numpy array (H, W, 3) in RGB
            
        Returns:
            Dict with:
                - status: 'Normal' | 'Anomaly'
                - confidence: float [0-1]
                - details: str
                - inference_time: float (seconds)
        """
        if self.model is None:
            logger.warning("⚠️ Vision model not loaded")
            return {
                "status": "Error",
                "confidence": 0.0,
                "details": "Vision model not available",
                "inference_time": 0.0
            }
        
        try:
            start_time = time.time()
            
            # Convert numpy array to PIL Image
            pil_image = Image.fromarray(image_array.astype('uint8'))
            
            # Apply transforms
            image_tensor = self.image_transform(pil_image).unsqueeze(0).to(self.device)
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(image_tensor)
            
            # Get predictions
            probabilities = torch.softmax(outputs, dim=1)[0]
            anomaly_prob = float(probabilities[1].cpu().numpy())  # Assuming index 1 is anomaly
            
            # Determine status
            status = "Anomaly" if anomaly_prob > 0.5 else "Normal"
            
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            self.inference_count += 1
            
            result = {
                "status": status,
                "confidence": anomaly_prob,
                "details": f"anomaly_prob={anomaly_prob:.3f}",
                "inference_time": inference_time
            }
            
            logger.debug(
                f"🎯 Vision inference #{self.inference_count}: "
                f"{status} (conf={anomaly_prob:.3f}, time={inference_time*1000:.1f}ms)"
            )
            
            return result
        
        except Exception as e:
            logger.error(f"❌ Vision inference failed: {e}")
            return {
                "status": "Error",
                "confidence": 0.0,
                "details": str(e),
                "inference_time": 0.0
            }
    
    def save_image(
        self,
        image_array: np.ndarray,
        timestamp: str,
        inference_result: Dict,
        filename: Optional[str] = None
    ) -> Optional[str]:
        """
        Save image to disk (if configured).
        
        Args:
            image_array: Image as numpy array
            timestamp: ISO timestamp string
            inference_result: Result from infer_image
            filename: Optional custom filename (without extension)
            
        Returns:
            Path to saved image, or None if not saved
        """
        # Check if persistence is enabled
        if not self.config.save_fallback_images:
            return None
        
        # Check if should only save anomalies
        if self.config.save_anomaly_images_only:
            if inference_result.get("status") != "Anomaly":
                return None
        
        try:
            # Create filename if not provided
            if filename is None:
                # Use timestamp as filename (sanitize for filesystem)
                filename = timestamp.replace(':', '-').replace('T', '_').split('.')[0]
            
            # Add status to filename
            status = inference_result.get("status", "unknown")
            filename = f"{filename}_{status}"
            
            # Construct full path
            save_dir = self.config.live_fallback_images_dir
            os.makedirs(save_dir, exist_ok=True)
            
            image_path = os.path.join(save_dir, f"{filename}.jpg")
            
            # Convert RGB to BGR for cv2
            image_bgr = cv2.cvtColor(image_array.astype('uint8'), cv2.COLOR_RGB2BGR)
            
            # Save image
            cv2.imwrite(
                image_path,
                image_bgr,
                [cv2.IMWRITE_JPEG_QUALITY, self.config.image_capture_quality]
            )
            
            logger.info(f"💾 Saved image: {image_path}")
            return image_path
        
        except Exception as e:
            logger.error(f"❌ Failed to save image: {e}")
            return None
    
    def run_fallback(
        self,
        sensor_inference_result: Dict
    ) -> Dict:
        """
        Run complete vision fallback pipeline.
        
        Args:
            sensor_inference_result: Result from sensor inference
            
        Returns:
            Dict with:
                - vision_status: 'Normal' | 'Anomaly' | 'Error'
                - vision_confidence: float
                - image_saved: bool
                - image_path: str or None
                - capture_time: float
                - inference_time: float
                - fallback_reason: str
        """
        result = {
            "vision_status": "Error",
            "vision_confidence": 0.0,
            "image_saved": False,
            "image_path": None,
            "capture_time": 0.0,
            "inference_time": 0.0,
            "fallback_reason": ""
        }
        
        # Determine fallback reason
        if sensor_inference_result.get("status") == "Sensor_Error":
            result["fallback_reason"] = "sensor_error"
        elif sensor_inference_result.get("confidence", 1.0) < self.config.confidence_threshold_for_vision_fallback:
            result["fallback_reason"] = "low_confidence"
        else:
            return result  # No fallback needed
        
        logger.info(f"🎥 Activating vision fallback ({result['fallback_reason']})...")
        
        try:
            # Capture image
            capture_start = time.time()
            capture_result = self.capture_image()
            capture_time = time.time() - capture_start
            result["capture_time"] = capture_time
            
            if capture_result is None:
                result["vision_status"] = "Capture_Failed"
                return result
            
            image_array, timestamp = capture_result
            
            # Run vision inference
            inference_result = self.infer_image(image_array)
            result["inference_time"] = inference_result.get("inference_time", 0.0)
            result["vision_status"] = inference_result.get("status", "Error")
            result["vision_confidence"] = inference_result.get("confidence", 0.0)
            
            # Save image (if configured)
            image_path = self.save_image(image_array, timestamp, inference_result)
            result["image_saved"] = image_path is not None
            result["image_path"] = image_path
            
            logger.info(
                f"✅ Vision fallback complete: {result['vision_status']} "
                f"(conf={result['vision_confidence']:.3f})"
            )
            
            return result
        
        except Exception as e:
            logger.error(f"❌ Vision fallback failed: {e}")
            result["vision_status"] = "Error"
            return result
    
    def get_stats(self) -> Dict:
        """Return vision inference statistics."""
        return {
            "inference_count": self.inference_count,
            "successful_captures": self.successful_captures,
            "failed_captures": self.failed_captures,
            "capture_success_rate": (
                self.successful_captures / (self.successful_captures + self.failed_captures)
                if (self.successful_captures + self.failed_captures) > 0 else 0
            ),
            "avg_inference_time_ms": (
                np.mean(self.inference_times) * 1000
                if self.inference_times else 0
            ),
        }
