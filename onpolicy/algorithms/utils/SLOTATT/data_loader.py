import os
import glob
import torch

from PIL import Image, ImageFile
from torchvision import transforms
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True


class GlobDataset(Dataset):
    def __init__(self, root, phase, img_glob='*.png'):
        self.root = root
        self.total_dirs = sorted(glob.glob(root))
        self.num_background_objects = 1
        self.input_channels = 3

        train_split = int(len(self.total_dirs) * 0.7)
        val_split = int(len(self.total_dirs) * 0.85)

        if phase == 'train':
            self.total_dirs = self.total_dirs[:train_split]
        elif phase == 'val':
            self.total_dirs = self.total_dirs[train_split:val_split]
        elif phase == 'test':
            self.total_dirs = self.total_dirs[val_split:]
        else:
            pass

        # chunk into episodes
        self.episodes = []
        for dir in self.total_dirs:
            image_paths = sorted(glob.glob(os.path.join(dir, img_glob)))
            for path in image_paths:
                data_tmp = torch.load(path)
                self.episodes.append(data_tmp)

        self.episodes = torch.cat(self.episodes, dim=0).permute(0, 3, 1, 2) / 255.0
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, idx):
        # (batch, channel, height, width)
        return {"image": self.episodes[idx]}
