"""Optional frozen MobileNetV3 comparison (ImageNet weights)."""
import numpy as np
from PIL import Image


class MobileNetFeatures:
    def __init__(self):
        import torch
        from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

        self.torch = torch
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        weights = MobileNet_V3_Small_Weights.DEFAULT
        self.transform = weights.transforms()
        self.model = mobilenet_v3_small(weights=weights).to(self.device).eval()
        self.model.classifier = torch.nn.Identity()
        probe = torch.zeros(1, 3, 224, 224, device=self.device)
        with torch.inference_mode():
            out = self.model(probe)
        assert out.device == self.device and torch.isfinite(out).all()
        print(f'MobileNet: torch={torch.__version__}, device={self.device}, output={out.device}', flush=True)

    def __call__(self, path):
        with Image.open(path) as im:
            tensor = self.transform(im.convert('RGB')).unsqueeze(0).to(self.device)
        with self.torch.inference_mode():
            return self.model(tensor).cpu().numpy().reshape(-1).astype(np.float64)
