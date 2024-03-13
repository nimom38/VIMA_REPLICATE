from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.utils.visualizer import Visualizer
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.evaluation import COCOEvaluator, inference_on_dataset

import cv2
import os
from typing import Literal
import pickle
import numpy as np
import random

random.seed(42)

def get_object_dicts(dir_path: str, view: Literal["front", "top"]):
  items = os.listdir(dir_path)
  task_dirs = [item for item in items if os.path.isdir(os.path.join(dir_path, item))]

  dataset_dicts = []

  for task_dir in task_dirs:
    items = os.listdir(os.path.join(dir_path, task_dir))
    trajectory_dirs = [item for item in items if os.path.isdir(os.path.join(dir_path, task_dir, item))]

    for trajectory_dir in trajectory_dirs:
      img_dir = f"rgb_{view}"
      img_dir_path = os.path.join(dir_path, task_dir, trajectory_dir, img_dir)
      img_files = os.listdir(img_dir_path)

      with open(os.path.join(dir_path, task_dir, trajectory_dir, "obs.pkl"), "rb") as f:
        obs = pickle.load(f)

      with open(os.path.join(dir_path, task_dir, trajectory_dir, "trajectory.pkl"), "rb") as f:
        traj_meta = pickle.load(f)

      all_segm = obs.pop("segm")[view]
      object_ids = traj_meta.pop("obj_id_to_info").keys()

      assert len(img_files) == all_segm.shape[0]

      for (img_file, segm) in zip(img_files, all_segm):
        record = {}

        img_file_path = os.path.join(img_dir_path, img_file)
        height, width = cv2.imread(img_file_path).shape[:2]

        assert height == segm.shape[0] and width == segm.shape[1]

        record["file_name"] = img_file_path
        record["image_id"] = os.path.join(task_dir, trajectory_dir, img_dir, img_file)
        record["height"] = height
        record["width"] = width

        objects = []

        for object_id in object_ids:
          ys, xs = np.nonzero(segm == object_id)
          if len(ys) < 2 or len(xs) < 2:
            continue
          y_min, y_max = np.min(ys), np.max(ys)
          x_min, x_max = np.min(xs), np.max(xs)

          object = {
            "bbox": [x_min, y_min, x_max, y_max],
            "bbox_mode": BoxMode.XYXY_ABS,
            "category_id": 0
          }
          objects.append(object)
        
        record["annotations"] = objects
        dataset_dicts.append(record)

  return dataset_dicts

def get_train_object_dicts(object_dicts):
  split_index = int(len(object_dicts) * 0.7)
  return object_dicts[:split_index]

def get_val_object_dicts(object_dicts):
  split_index = int(len(object_dicts) * 0.7)
  return object_dicts[split_index:]

def train_obj_detection(cfg):
  cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
  cfg.SOLVER.IMS_PER_BATCH = 2 #####
  cfg.SOLVER.BASE_LR = 0.0005 #####
  cfg.SOLVER.MAX_ITER = 150 #####
  cfg.SOLVER.STEPS = []
  cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512 #####
  cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
  cfg.MODEL.MASK_ON = False
  trainer = DefaultTrainer(cfg)
  trainer.resume_or_load(resume=True)
  trainer.train()

def val_obj_detection(cfg):
  cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
  print("PATH XYZ", os.path.join(cfg.OUTPUT_DIR, "model_final.pth"))
  print("config XYZ", cfg.SOLVER.IMS_PER_BATCH)
  cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
  predictor = DefaultPredictor(cfg)

  model, aug = predictor.model, predictor.aug
  print(model)
  print(aug)

  # for idx, d in enumerate(val_object_dicts):
  #   if idx == 5:
  #     break
  #   img = cv2.imread(d["file_name"])
  #   visualizer = Visualizer(img[:, :, ::-1], metadata=table_object_metadata, scale=3)
  #   out = visualizer.draw_dataset_dict(d)
  #   cv2.imshow(d["image_id"], out.get_image()[:, :, ::-1])

  # evaluator = COCOEvaluator("val", output_dir="./output")
  # val_loader = build_detection_test_loader(cfg, "val")
  # print(inference_on_dataset(predictor.model, val_loader, evaluator))

if __name__ == "__main__":
  dataset_dir = "/Users/nimom/Documents/AI_Research/VIMA_Recreate/archive/mini_dataset"
  object_dicts = get_object_dicts(dataset_dir, "front")

  random.shuffle(object_dicts)

  DatasetCatalog.register("train", lambda : get_train_object_dicts(object_dicts))
  MetadataCatalog.get("train").set(thing_classes=["table_object"])
  DatasetCatalog.register("val", lambda: get_val_object_dicts(object_dicts))
  MetadataCatalog.get("val").set(thing_classes=["table_object"])

  table_object_metadata = MetadataCatalog.get("train")

  train_object_dicts = get_train_object_dicts(object_dicts)
  val_object_dicts = get_val_object_dicts(object_dicts)

  cfg = get_cfg()
  cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
  print(cfg)
  cfg.DATASETS.TRAIN = ("train")
  cfg.DATASETS.TEST = ()
  cfg.MODEL.DEVICE = "cpu"

  # train_obj_detection(cfg)
  # print("Training complete!")

  val_obj_detection(cfg)
  # cv2.waitKey(0)

