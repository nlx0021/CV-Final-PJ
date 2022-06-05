from random import Random
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader, random_split, sampler

class PartialDataset(Dataset):
    def __init__(self, dataset, n_items=10):
        self.dataset = dataset
        self.n_items = n_items

    def __getitem__(self, idx):
        return self.dataset.__getitem__(idx)

    def __len__(self):
        return min(self.n_items, len(self.dataset))
    

def get_cifar(root,
              train,
              transform,
              n_items):
    dataset = datasets.CIFAR100(root=root,
                               train=train,
                               download=True,
                               transform=transform)   
    if n_items > 0:
        dataset = PartialDataset(dataset, n_items)
    return dataset