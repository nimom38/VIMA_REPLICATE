import torch
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.modeling import GeneralizedRCNN, META_ARCH_REGISTRY

@META_ARCH_REGISTRY.register()
class VIMADetectronRCNN(GeneralizedRCNN):
  def __init__(self, cfg):
    super().__init__(cfg)

ckpt_path = "/Users/nimom/Documents/AI_Research/VIMA_Recreate/archive/models/mask_rcnn.pth"
ckpt = torch.load(ckpt_path, map_location="cpu")

ckpt_cfg = ckpt.pop("cfg")

cfg = get_cfg()
cfg.update(**ckpt_cfg)
cfg.MODEL.WEIGHTS = ckpt_path
cfg.MODEL.DEVICE = "cpu"

predictor = DefaultPredictor(cfg)
model, aug = predictor.model, predictor.aug