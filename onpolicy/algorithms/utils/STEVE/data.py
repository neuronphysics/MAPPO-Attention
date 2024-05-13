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
            frame_buffer = []
            image_paths = sorted(glob.glob(os.path.join(dir, img_glob)))
            for path in image_paths:
                frame_buffer.append(path)
                if len(frame_buffer) == self.ep_len:
                    self.episodes.append(frame_buffer)
                    frame_buffer = []

        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, idx):
        video = []
        for img_loc in self.episodes[idx]:
            image = Image.open(img_loc).convert("RGB")
            tensor_image = self.transform(image)
            video += [tensor_image]
        video = torch.stack(video, dim=0)
        return video
