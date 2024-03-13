from .dataset import TrainSet

cache_dir = '/Users/nimom/Documents/AI_Research/VIMA_Recreate/cache'
dataset_dir = '/Users/nimom/Documents/AI_Research/VIMA_Recreate/archive/mini_dataset'

dataset = TrainSet(cache_dir=cache_dir, dataset_dir=dataset_dir)

print(dataset[1])