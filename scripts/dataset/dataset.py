import os
import cv2
import pickle
from typing import Literal

from einops import rearrange

from object_detector import predictor

import torch
from torch.utils.data import Dataset

class TrainSet(Dataset):
  cache_dir: str
  dataset_size: int = 0
  seq_pad: int

  def get_imgs(self, img, bboxes):
    imgs = []
    for bbox in bboxes:
      x_min = int(bbox[0].item())
      y_min = int(bbox[1].item())
      x_max = int(bbox[2].item())
      y_max = int(bbox[3].item())
      imgs.append(img[y_min:y_max+1, x_min:x_max+1])
    return imgs

  def resize_img(self, img):
    height_pad, width_pad = max(img.shape[1]-img.shape[0], 0), max(img.shape[0]-img.shape[1], 0)
    top_pad, bottom_pad = height_pad // 2, height_pad - (height_pad // 2)
    left_pad, right_pad = width_pad // 2, width_pad - (width_pad // 2)
    
    padded_image = cv2.copyMakeBorder(img, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return cv2.resize(padded_image, (32, 32), interpolation=cv2.INTER_AREA)

  def get_obs_imgs_and_bboxes(self, dir: str):
    img_list = os.listdir(dir)
    all_bboxes = []
    all_imgs = []
    for img in img_list:
      img_file = cv2.imread(os.path.join(dir, img))
      bboxes = predictor(img_file)["instances"].pred_boxes.tensor
      all_bboxes.append(bboxes)
      imgs = self.get_imgs(img_file, bboxes)
      imgs = [self.resize_img(img) for img in imgs]
      all_imgs.append(imgs)

    return all_imgs, all_bboxes
  
  def get_prompt_imgs_and_bboxes(self, prompt_assets: dict, view: Literal["front", "top"]):
    all_bboxes = []
    all_imgs = []
    for k, v in prompt_assets.items():
      img = rearrange(v["rgb"][view], "c h w -> h w c")
      img = img[:, :, ::-1]
      bboxes = predictor(img)["instances"].pred_boxes.tensor
      imgs = self.get_imgs(img, bboxes)
      imgs = [self.resize_img(img) for img in imgs]
      all_imgs.append(imgs)
      if v["placeholder_type"] == "scene":
        all_bboxes.append(bboxes)
      else:
        all_bboxes.append([torch.tensor([0, 0, 0, 0])])

    return all_imgs, all_bboxes

  def __init__(self, *, cache_dir: str, dataset_dir: str):
    self.cache_dir = cache_dir

    if os.path.exists(cache_dir):
      return
    
    self.dataset_size = 0
    os.makedirs(os.path.join(cache_dir, "dataset"))

    task_list = os.listdir(dataset_dir)
    for task in task_list:
      trajectories = os.listdir(os.path.join(dataset_dir, task))
      for trajectory in trajectories:
        if trajectory in ['.DS_Store', 'metadata.pkl']:
          continue
        mask_rotate = True
        if trajectory in ["rotate", "twist"]:
          mask_rotate = False

        front_obs_imgs, front_obs_bboxes = self.get_obs_imgs_and_bboxes(os.path.join(dataset_dir, task, trajectory, "rgb_front"))
        top_obs_imgs, top_obs_bboxes = self.get_obs_imgs_and_bboxes(os.path.join(dataset_dir, task, trajectory, "rgb_top"))

        obs_imgs = []
        for (front_obs, top_obs) in zip(front_obs_imgs, top_obs_imgs):
          temp = []
          for (front_img, top_img) in zip(front_obs, top_obs):
            temp.append(front_img)
            temp.append(top_img)
          obs_imgs.append(temp)

        obs_bboxes = []
        for (front_obs, top_obs) in zip(front_obs_bboxes, top_obs_bboxes):
          temp = []
          for (front_bbox, top_bbox) in zip(front_obs, top_obs):
            temp.append(front_bbox)
            temp.append(top_bbox)
          obs_bboxes.append(temp)

        with open(os.path.join(dataset_dir, task, trajectory, "action.pkl"), "rb") as f:
          action = pickle.load(f)
        pose0_position = action["pose0_position"]
        pose0_rotation = action["pose0_rotation"]
        pose1_position = action["pose1_position"]
        pose1_rotation = action["pose1_rotation"]

        pose0 = [(pos, rot) for (pos, rot) in zip(pose0_position, pose0_rotation)]
        pose1 = [(pos, rot) for (pos, rot) in zip(pose1_position, pose1_rotation)]
        actions = [(p0, p1) for (p0, p1) in zip(pose0, pose1)]

        with open(os.path.join(dataset_dir, task, trajectory, "obs.pkl"), "rb") as f:
          obs = pickle.load(f)
        end_effectors = obs.pop("ee")

        with open(os.path.join(dataset_dir, task, trajectory, "trajectory.pkl"), "rb") as f:
          traj_meta = pickle.load(f)
        prompt, prompt_assets = traj_meta.pop("prompt"), traj_meta.pop("prompt_assets")

        front_prompt_imgs, front_prompt_bboxes = self.get_prompt_imgs_and_bboxes(prompt_assets, "front")
        top_prompt_imgs, top_prompt_bboxes = self.get_prompt_imgs_and_bboxes(prompt_assets, "top")

        prompt_imgs = []
        for (front_prompt, top_prompt) in zip(front_prompt_imgs, top_prompt_imgs):
          temp = []
          for (front_img, top_img) in zip(front_prompt, top_prompt):
            temp.append(front_img)
            temp.append(top_img)
          prompt_imgs.append(temp)

        prompt_bboxes = []
        for (front_prompt, top_prompt) in zip(front_prompt_bboxes, top_prompt_bboxes):
          temp = []
          for (front_bbox, top_bbox) in zip(front_prompt, top_prompt):
            temp.append(front_bbox)
            temp.append(top_bbox)
          prompt_bboxes.append(temp)

        instance = {
          "prompt_sentence": prompt,
          "prompt_imgs": prompt_imgs,
          "prompt_bboxes": prompt_bboxes,
          "obs_imgs": obs_imgs,
          "obs_bboxes": obs_bboxes,
          "end_effectors": end_effectors,
          "action": actions,
          "mask_rotate": mask_rotate,
        }

        with open(os.path.join(cache_dir, "dataset", f"{self.dataset_size}.pkl"), "wb") as f:
          pickle.dump(instance, f)
          self.dataset_size += 1
          if self.dataset_size%10 == 0:
            print(f"{self.dataset_size} instances completed!")

  def __len__(self):
    return self.dataset_size
  
  def __getitem__(self, idx):
    filepath = os.path.join(self.cache_dir, "dataset", f"{idx}.pkl")
    with open(filepath, "rb") as f:
      instance = pickle.load(f)
    return instance

