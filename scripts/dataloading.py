from torch.utils.data import DataLoader, random_split
from scripts.dataset import TrainSet

n_train = 600000
n_val = 50000

cache_dir = '/Users/nimom/Documents/AI_Research/VIMA_Recreate/cache'
dataset_dir = '/Users/nimom/Documents/AI_Research/VIMA_Recreate/archive/mini_dataset'

dataset = TrainSet(cache_dir=cache_dir, dataset_dir=dataset_dir)

assert n_train+n_val == len(dataset), "Split sizes does not sum to dataset size!"

train_dataset, val_dataset = random_split(dataset, [n_train, n_val])


seq_pad = dataset.seq_pad # change
dummy_vit_pad = 10 # change
def collate_fn(batch):
  
  pass


batch_size = 64
num_workers = 4
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)
val_loaderc = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)

