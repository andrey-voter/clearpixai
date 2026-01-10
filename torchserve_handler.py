"""TorchServe handler for watermark detection model.

This handler implements pre- and post-processing for the watermark detection model.
It accepts image data and returns a binary mask indicating watermark regions.
"""

import base64
import io
import json
import logging
from typing import Dict, List

import numpy as np
import torch
from PIL import Image
import segmentation_models_pytorch as smp

logger = logging.getLogger(__name__)


class WatermarkDetectionHandler:
    """Handler for watermark detection model in TorchServe."""
    
    def __init__(self):
        """Initialize handler."""
        self.model = None
        self.device = None
        self.preprocessing_fn = None
        self.image_size = None
        self.threshold = 0.5
        self.encoder_name = "mit_b5"
    
    def initialize(self, context):
        """Initialize model and preprocessing.
        
        Args:
            context: TorchServe context containing model files and properties
        """
        properties = context.system_properties
        model_dir = properties.get("model_dir")
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        
        logger.info(f"Loading model from {model_dir} on device {self.device}")
        
        # Load model file
        model_pt = None
        for file in context.manifest["model"]["modelFiles"]:
            if file.endswith(".pt"):
                model_pt = file
                break
        
        if model_pt is None:
            raise RuntimeError("Model file (.pt) not found in model archive")
        
        model_path = f"{model_dir}/{model_pt}"
        
        # Load TorchScript model
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.eval()
        self.model.to(self.device)
        
        logger.info(f"Model loaded successfully on {self.device}")
        
        # Load config if available
        try:
            config_path = f"{model_dir}/config.json"
            with open(config_path, 'r') as f:
                config = json.load(f)
                self.encoder_name = config.get("encoder_name", "mit_b5")
                self.image_size = config.get("image_size", 512)
                self.threshold = config.get("threshold", 0.5)
        except Exception as e:
            logger.warning(f"Could not load config.json: {e}. Using defaults.")
            self.image_size = 512
            self.threshold = 0.5
        
        # Get preprocessing function for encoder
        try:
            self.preprocessing_fn = smp.encoders.get_preprocessing_fn(
                self.encoder_name, "imagenet"
            )
        except Exception as e:
            logger.warning(f"Could not load preprocessing function: {e}")
            self.preprocessing_fn = None
        
        logger.info(
            f"Handler initialized: encoder={self.encoder_name}, "
            f"image_size={self.image_size}, threshold={self.threshold}"
        )
    
    def preprocess(self, requests: List[Dict]) -> tuple:
        """Preprocess input images.
        
        Args:
            requests: List of request dictionaries containing image data
            
        Returns:
            Tuple of (preprocessed image tensor [batch_size, 3, H, W], original_sizes)
        """
        images = []
        original_sizes = []
        
        for request in requests:
            # Handle different input formats
            if isinstance(request, dict):
                # JSON request with base64 image
                if "body" in request:
                    body = request["body"]
                    if isinstance(body, str):
                        body = json.loads(body)
                    
                    if "image" in body:
                        # Base64 encoded image
                        image_data = body["image"]
                        if isinstance(image_data, str):
                            image_bytes = base64.b64decode(image_data)
                            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                        else:
                            raise ValueError("Invalid image format in request body")
                    elif "image_path" in body:
                        # Path to image file
                        image_path = body["image_path"]
                        image = Image.open(image_path).convert("RGB")
                    else:
                        raise ValueError("No image data found in request body")
                elif "data" in request:
                    # Raw image bytes
                    image_bytes = request["data"]
                    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                else:
                    raise ValueError("Invalid request format")
            else:
                # Raw bytes
                image = Image.open(io.BytesIO(request)).convert("RGB")
            
            # Store original size
            original_size = image.size
            original_sizes.append(original_size)
            
            # Resize if needed
            if self.image_size:
                image = image.resize(
                    (self.image_size, self.image_size),
                    Image.BICUBIC
                )
            
            # Convert to numpy array
            np_image = np.array(image).astype("float32")
            
            # Apply preprocessing
            if self.preprocessing_fn:
                np_image = self.preprocessing_fn(np_image)
            else:
                # Default normalization
                np_image = np_image / 255.0
            
            # Convert to tensor: [H, W, C] -> [C, H, W]
            tensor = torch.from_numpy(np_image).permute(2, 0, 1)
            images.append(tensor)
        
        # Stack into batch
        batch_tensor = torch.stack(images).to(self.device, dtype=torch.float32)
        
        return batch_tensor, original_sizes
    
    def inference(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Run model inference.
        
        Args:
            input_tensor: Preprocessed image tensor [batch_size, 3, H, W]
            
        Returns:
            Model logits [batch_size, 1, H, W]
        """
        with torch.no_grad():
            logits = self.model(input_tensor)
        return logits
    
    def postprocess(self, logits: torch.Tensor, original_sizes: List[tuple] = None) -> List[Dict]:
        """Postprocess model output.
        
        Args:
            logits: Model logits [batch_size, 1, H, W]
            original_sizes: List of original image sizes (width, height)
            
        Returns:
            List of response dictionaries containing mask data
        """
        batch_size = logits.shape[0]
        responses = []
        
        # Apply sigmoid
        probs = torch.sigmoid(logits)
        
        for i in range(batch_size):
            prob_mask = probs[i, 0].cpu().numpy()  # [H, W]
            
            # Resize to original size if needed
            if original_sizes and i < len(original_sizes):
                original_size = original_sizes[i]
                if self.image_size and (
                    original_size[0] != self.image_size or
                    original_size[1] != self.image_size
                ):
                    # Resize mask back to original size
                    mask_image = Image.fromarray((prob_mask * 255).astype(np.uint8))
                    mask_image = mask_image.resize(original_size, Image.BILINEAR)
                    prob_mask = np.array(mask_image) / 255.0
            
            # Create binary mask
            binary_mask = (prob_mask >= self.threshold).astype(np.uint8)
            
            # Calculate statistics
            watermark_ratio = binary_mask.mean()
            max_confidence = float(prob_mask.max())
            
            # Convert mask to base64 for JSON response
            mask_image = Image.fromarray(binary_mask * 255, mode="L")
            buffer = io.BytesIO()
            mask_image.save(buffer, format="PNG")
            mask_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
            
            response = {
                "mask": mask_base64,
                "watermark_ratio": float(watermark_ratio),
                "max_confidence": max_confidence,
                "threshold": self.threshold,
                "mask_shape": list(binary_mask.shape),
            }
            
            responses.append(response)
        
        return responses
    
    def handle(self, data, context):
        """Main handler method called by TorchServe.
        
        Args:
            data: Input data (list of requests)
            context: TorchServe context
            
        Returns:
            List of response dictionaries
        """
        # Preprocess (returns tensor and original sizes)
        input_tensor, original_sizes = self.preprocess(data)
        
        # Inference
        logits = self.inference(input_tensor)
        
        # Postprocess
        responses = self.postprocess(logits, original_sizes)
        
        return responses


# TorchServe entry point
_service = None


def handle(data, context):
    """TorchServe entry point."""
    global _service
    
    if _service is None:
        _service = WatermarkDetectionHandler()
        _service.initialize(context)
    
    return _service.handle(data, context)

