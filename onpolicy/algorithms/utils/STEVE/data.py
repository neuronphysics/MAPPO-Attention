import os
import glob
import torch

from PIL import Image, ImageFile
from torchvision import transforms
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True


class GlobVideoDataset(Dataset):
    def __init__(self, root, phase, ep_len=3, img_glob='*.png'):
        self.root = root
        self.total_dirs = sorted(glob.glob(root))
        self.ep_len = ep_len

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
                frame_buffer = []
                data_tmp = torch.load(path)
                for i in range(data_tmp.shape[0]):
                    frame_buffer.append(data_tmp[i])
                    if (i + 1) % self.ep_len == 0:
                        self.episodes.append(torch.stack(frame_buffer, 0))
                        frame_buffer = []

        self.episodes = torch.stack(self.episodes, dim=0).permute(0, 1, 4, 2, 3) / 255.0
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, idx):
        # (batch, seq, channel, height, width)
        return self.episodes[idx]
