import os
import glob
import torch
import random
from PIL import Image, ImageFile
from torchvision import transforms
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_pil_image, InterpolationMode
from torchvision.transforms import Resize

ImageFile.LOAD_TRUNCATED_IMAGES = True


class GlobDataset(Dataset):
    def __init__(self, root, phase, crop_repeat=3, img_glob='*.pt'):
        self.root = root
        self.total_dirs = sorted(glob.glob(root))
        self.num_background_objects = 1
        self.input_channels = 3
        self.crop_size = 44
        self.crop_repeat = crop_repeat

        self.resize = Resize((224, 224), interpolation=InterpolationMode.BILINEAR)

        train_split = int(len(self.total_dirs) * 0.8)
        val_split = int(len(self.total_dirs) * 0.95)

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
                data_tmp = torch.load(path).permute(0, 3, 1, 2)
                _, _, h, w = data_tmp.shape
                if h <= self.crop_size or w <= self.crop_size:
                    continue
                for t in data_tmp:
                    for i in range(self.crop_repeat):
                        cropped_img = self.random_crop_img(t)
                        # resize it to 224
                        resized_img = self.resize(cropped_img)
                        self.episodes.append(resized_img / 255.0)

        self.episodes = torch.stack(self.episodes, dim=0)

        self.transform = transforms.ToTensor()

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
