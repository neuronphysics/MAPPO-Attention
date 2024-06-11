import os
import glob
import torch
import random
from PIL import Image, ImageFile
from torchvision import transforms
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True


class GlobDataset(Dataset):
    def __init__(self, agent_root="", world_root="", phase="train", crop_repeat=3, img_glob='*.png', crop_size=44):
        self.agent_root = agent_root
        self.world_root = world_root
        self.agent_total_dirs = sorted(glob.glob(agent_root))
        self.world_total_dirs = sorted(glob.glob(world_root))
        self.num_background_objects = 1
        self.input_channels = 3
        self.crop_size = crop_size
        self.crop_repeat = crop_repeat

        train_split_percent = 0.75
        self.episodes = []
        if agent_root is not None and agent_root != "":
            train_split = int(len(self.agent_total_dirs) * train_split_percent)
            val_split = int(len(self.agent_total_dirs))

            if phase == 'train':
                self.agent_total_dirs = self.agent_total_dirs[:train_split]
            elif phase == 'val':
                self.agent_total_dirs = self.agent_total_dirs[train_split:val_split]
            else:
                pass

            # chunk into episodes
            for dir in self.agent_total_dirs:
                image_paths = sorted(glob.glob(os.path.join(dir, img_glob)))
                for path in image_paths:
                    data_tmp = torch.load(path).permute(0, 3, 1, 2)
                    self.episodes.extend(data_tmp / 255.0)

        if world_root is not None and world_root != "":
            train_split = int(len(self.world_total_dirs) * train_split_percent)
            val_split = int(len(self.world_total_dirs))

            if phase == 'train':
                self.world_total_dirs = self.world_total_dirs[:train_split]
            elif phase == 'val':
                self.world_total_dirs = self.world_total_dirs[train_split:val_split]
            else:
                pass

            for dir in self.world_total_dirs:
                image_paths = sorted(glob.glob(os.path.join(dir, img_glob)))
                for path in image_paths:
                    data_tmp = torch.load(path).permute(0, 3, 1, 2)
                    _, _, h, w = data_tmp.shape
                    if h <= self.crop_size or w <= self.crop_size:
                        continue
                    for t in data_tmp:
                        for i in range(self.crop_repeat):
                            cropped_img = self.random_crop_img(t)
                            self.episodes.append(cropped_img / 255.0)

        self.episodes = torch.stack(self.episodes, dim=0)

    def random_crop_img(self, img):
        max_x = img.shape[2] - self.crop_size
        max_y = img.shape[1] - self.crop_size

        x = random.randint(0, max_x)
        y = random.randint(0, max_y)

        crop_img = img[:, y:y + self.crop_size, x:x + self.crop_size]
        return crop_img

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, idx):
        # (batch, channel, height, width)
        return self.episodes[idx]
