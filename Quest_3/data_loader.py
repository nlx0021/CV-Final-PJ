import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchtoolbox.transform import Cutout
import torchvision.datasets as datasets
import torch

from dataset import get_cifar


def get_cifar_loader(root='./data/',
                     batch_size=128,
                     shuffle=True,
                     num_workers=4,
                     n_items=512,
                     data_aug=True,
                     cutout=False):
    
    # Transforms.
    normalize = transforms.Normalize(mean=[125.3/255., 123./255., 113.9/255.],
                                     std=[63./255., 62.1/255., 66.7/255.])

    data_transforms = transforms.Compose(
        [transforms.ToTensor(),
        normalize])
    
    if cutout:
        data_aug_transforms = transforms.Compose(
        [Cutout(p=.5, scale=(.1, .3), ratio=(.8, 1/.8), value=(0, 255)),
         transforms.RandomOrder(
             [transforms.RandomResizedCrop(size=32, scale=(.7, 1), ratio=(4/5, 5/4)),
              transforms.RandomHorizontalFlip()
              # transforms.RandomVerticalFlip(),
              ]
            ),
         transforms.ToTensor(),
        normalize])
    
    else:
        data_aug_transforms = transforms.Compose(
            [
            transforms.RandomOrder(
                [transforms.RandomResizedCrop(size=32, scale=(.7, 1), ratio=(4/5, 5/4)),
                transforms.RandomHorizontalFlip()
                # transforms.RandomVerticalFlip(),
                ]
                ),
            transforms.ToTensor(),
            normalize])
        
    # Get the dataset.
    dataset = get_cifar(root=root, train=True, transform=data_transforms, n_items=n_items)
    test_dataset = get_cifar(root=root, train=False, transform=data_transforms, n_items=n_items)
    size = len(dataset)
    if not data_aug:
        train_dataset, val_dataset = random_split(dataset, [size - size//10, size//10], generator=torch.Generator().manual_seed(21))
    else:
        aug_dataset = get_cifar(root=root, train=True, transform=data_aug_transforms, n_items=n_items)
        # Set the seed to get the same split.
        _, val_dataset = random_split(dataset, [size - size//10, size//10], generator=torch.Generator().manual_seed(21))
        train_dataset, _ = random_split(aug_dataset, [size - size//10, size//10], generator=torch.Generator().manual_seed(21))

    # Get the data loader.
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)    

    return train_loader, val_loader, test_loader



if __name__ == '__main__':
    # _, _, test_loader = get_cifar_loader(n_items=-1)
    # for X, y in test_loader:
    #     print(X[0])
    #     print(y[0])
    #     print(X[0].shape)
    #     img = np.transpose(X[0], [1,2,0])
    #     plt.imshow(img*0.5 + 0.5)
    #     plt.show()
    #     print(X[0].max())
    #     print(X[0].min())
    #     break
    
    train_loader, _, _ = get_cifar_loader(n_items=512, cutout=True)
    for X, y in train_loader:
        print(X[0])
        print(y[0])
        print(X[0].shape)
        img = np.transpose(X[0], [1,2,0])
        plt.imshow(img*0.5 + 0.5)
        plt.show()
        print(X[0].max())
        print(X[0].min())
        