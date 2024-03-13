import cv2
from dataset import TrainSet

cache_dir = '/Users/nimom/Documents/AI_Research/VIMA_Recreate/cache'
dataset_dir = '/Users/nimom/Documents/AI_Research/VIMA_Recreate/archive/mini_dataset'

dataset = TrainSet(cache_dir=cache_dir, dataset_dir=dataset_dir)

instance = dataset[1]

print(instance["prompt_sentence"])
print(len(instance["obs_imgs"]))

id = 0
for obs in instance["obs_bboxes"]:
  print(len(obs))
  for bbox in obs:
    print(bbox)
    id += 1
