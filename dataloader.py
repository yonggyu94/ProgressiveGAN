import os
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as Transforms


class CustomDataset(Dataset):
    def __init__(self, root, transform):
        super(CustomDataset, self).__init__()
        self.root = root
        self.img_list = os.listdir(root)
        self.transform = transform

    def __getitem__(self, item):
        img_root = os.path.join(self.root, self.img_list[item])
        img = Image.open(img_root)
        tensor_img = self.transform(img)
        return tensor_img

    def __len__(self):
        return len(self.img_list)


def data_loader(config):
    transform = []
    transfrom.append(Transforms.RandomHorizontalFlip())
    transfrom.append(Transforms.ToTensor())
    transfrom.append(Transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = Transforms.Compose(transform)

    dataset = CustomDataset(config.data_root, transform=transform)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=8)
    return loader
