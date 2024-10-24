import os
import glob
import torch
import random
from PIL import Image, ImageFile
from torchvision import transforms
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True


class GlobDataset(Dataset):
    def __init__(self, agent_root="", world_root="", phase="train", crop_repeat=3, img_glob='*.pt', crop_size=44, data_size=None):
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
        
        # Calculate the size of the world dataset
        world_size = 0 if data_size is None else data_size

        # Process world data
        if world_root is not None and world_root != "":
            train_split = int(len(self.world_total_dirs) * train_split_percent)
            val_split = len(self.world_total_dirs)

            if phase == 'train':
                self.world_total_dirs = self.world_total_dirs[:train_split]
            elif phase == 'val':
                self.world_total_dirs = self.world_total_dirs[train_split:val_split]
            else:
                pass

            for dir in self.world_total_dirs:
                image_paths = sorted(glob.glob(os.path.join(dir, img_glob)))
                for path in image_paths:
                    data_tmp = torch.load(path)
                    world_size += len(data_tmp) * self.crop_repeat
                    data_tmp = data_tmp.permute(0, 3, 1, 2)
                    _, _, h, w = data_tmp.shape
                    if h < self.crop_size or w < self.crop_size:
                        continue

                    for t in data_tmp:
                        for _ in range(self.crop_repeat):
                            cropped_img = self.random_crop_img(t)
                            self.episodes.append(cropped_img / 255.0)
        
        self.data_size = world_size
        
        # Process agent data with dynamic sampling to match world size
        if agent_root is not None and agent_root != "":
            train_split = int(len(self.agent_total_dirs) * train_split_percent)
            val_split = len(self.agent_total_dirs)
            
            if phase == 'train':
                self.agent_total_dirs = self.agent_total_dirs[:train_split]
            elif phase == 'val':
                self.agent_total_dirs = self.agent_total_dirs[train_split:val_split]
            else:
                pass
            
            total_agent_dirs = len(self.agent_total_dirs)
            
            if total_agent_dirs > 0 and world_size > 0:
                target_rows_per_dir = world_size // total_agent_dirs
                remaining_rows = world_size

                print(f"Target rows per agent directory: {target_rows_per_dir}")
                
                for i, dir in enumerate(self.agent_total_dirs):
                    dir_episodes = []
                    image_paths = sorted(glob.glob(os.path.join(dir, img_glob)))

                    # Calculate total rows in this directory
                    total_rows = sum(len(torch.load(path)) for path in image_paths)
                    
                    # Calculate sampling ratio for this directory
                    if i == total_agent_dirs - 1:
                        sampling_ratio = remaining_rows / total_rows if total_rows > 0 else 0
                    else:
                        sampling_ratio = min(1.0, target_rows_per_dir / total_rows) if total_rows > 0 else 0
                    
                    # Load and sample data
                    for path in image_paths:
                        data_tmp = torch.load(path)
                        data_tmp = self.filter_black_images(data_tmp)  # Filter out black images
                        agent_data_size = len(data_tmp)
                        data_tmp = data_tmp.permute(0, 3, 1, 2)
                        num_samples = min(agent_data_size, max(1, int(agent_data_size * sampling_ratio)))
                        if agent_data_size> 0:
                        
                           indices = random.sample(range(agent_data_size), num_samples)
                           sampled_data = data_tmp[indices]
                           dir_episodes.extend(sampled_data / 255.0)
                    
                    # Further sample if we got too many episodes
                    if len(dir_episodes) > target_rows_per_dir:
                        dir_episodes = random.sample(dir_episodes, target_rows_per_dir)
                        
                    remaining_rows -= len(dir_episodes)
                    self.episodes.extend(dir_episodes)
                    print(f"Directory {dir}: sampled {len(dir_episodes)} rows")
        
        # Convert to tensor only after sampling to avoid memory spikes
        if len(self.episodes) > 0:
            self.episodes = torch.stack(self.episodes, dim=0)
        
        print(f"Final dataset size: {len(self.episodes)} rows")
        print(f"Memory usage: {self.episodes.element_size() * self.episodes.nelement() / (1024**3):.2f} GB")

    def get_size_info(self):
        """
        Returns information about the dataset size
        """
        return {
            'total_rows': len(self.episodes),
            'memory_gb': self.episodes.element_size() * self.episodes.nelement() / (1024**3) if len(self.episodes) > 0 else 0,
            'shape': self.episodes.shape if len(self.episodes) > 0 else None
        }
    def filter_black_images(self, batch, threshold=1e-2):
        """
        Filters out 'almost black' images from a batch.
        
        Args:
            batch (torch.Tensor): Batch of images, expected shape [batch_size, H, W, C].
            threshold (float): Pixel intensity threshold to identify black images. Default is 1e-3.
        
        Returns:
            torch.Tensor: Batch of images with black images excluded.
        """
        filtered_images = []
        total_images = len(batch)
        for img in batch:
            if not torch.all(img.abs() <= threshold):  # Check if the image is not black
                filtered_images.append(img)

        if filtered_images:
            print(f"Filtered out {len(filtered_images)}/{total_images} as non black images")
            return torch.stack(filtered_images)
        else:
            raise ValueError("agent data is all black")   # Return empty tensor if no valid images
        
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