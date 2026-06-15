from torchvision.datasets import OxfordIIITPet
from torchvision import transforms
from torch.utils.data import DataLoader


# processes each image from dataset
def get_transforms(augment:bool = False):
    if augment:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            #random transformation of data := augmentation
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15), # change if to fine tune
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                #saturation=0.2,
            ),
            transforms.ToTensor(),
            # normalize picture as it would a random data
            # like first step in max pool
            transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
        ])
    if not augment:
        return transforms.Compose([
            #should be same as in previous case
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
        ])
    else:
        raise TypeError('augmentation not implemented')


def get_dataloaders(batch_size:int=32, augment:bool=False):
    transforms = get_transforms(augment)

    train_dataset = OxfordIIITPet(
        root="./data",
        split='trainval',
        target_types="category",
        download=True,
        transform=transforms
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, test_loader































