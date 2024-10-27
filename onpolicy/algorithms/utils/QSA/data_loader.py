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
        self.episodes = []
        
        # Calculate the size of the world dataset
        world_size = 0 if data_size is None else data_size

        # Process world data
        if world_root is not None and world_root != "":
            train_split_percent = 0.75
            train_split = int(len(self.world_total_dirs) * train_split_percent)
            val_split = len(self.world_total_dirs)

            if phase == 'train':
                self.world_total_dirs = self.world_total_dirs[:train_split]
            elif phase == 'val':
                self.world_total_dirs = self.world_total_dirs[train_split:val_split]

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
        print(f"World data size: {world_size}")
        
        # Process agent data
        if agent_root is not None and agent_root != "" and world_size > 0:
            train_split_percent = 1.0
            train_split = int(len(self.agent_total_dirs) * train_split_percent)
            val_split = len(self.agent_total_dirs)
            
            if phase == 'train':
                self.agent_total_dirs = self.agent_total_dirs[:train_split]
            elif phase == 'val':
                self.agent_total_dirs = self.agent_total_dirs[train_split:val_split]
            
            if len(self.agent_total_dirs) > 0:
                # Clear episodes list to replace world data with agent data
                self.episodes = []
                remaining_rows = world_size
                used_dirs = set()
                
                # Initialize set of unused directories
                unused_dirs = set(self.agent_total_dirs)
                
                print(f"Target agent data size: {world_size}")
                print(f"Available directories: {len(unused_dirs)}")
                
                while remaining_rows > 0 and unused_dirs:
                    # Randomly select a directory from unused ones
                    dir = random.choice(list(unused_dirs))
                    unused_dirs.remove(dir)  # Remove it from unused set
                    
                    print(f"Processing directory: {dir}")
                    dir_episodes = []
                    image_paths = sorted(glob.glob(os.path.join(dir, img_glob)))
                    
                    # Load and filter all valid images from this directory
                    for path in image_paths:
                        try:
                            data_tmp = torch.load(path)
                            data_tmp = self.filter_black_images(data_tmp)
                            data_tmp = data_tmp.permute(0, 3, 1, 2)
                            dir_episodes.extend([img / 255.0 for img in data_tmp])
                        except ValueError as e:
                            print(f"Skipping file {path}: {str(e)}")
                            continue
                    
                    if dir_episodes:
                        # Take as many samples as we can up to remaining_rows
                        num_samples = min(len(dir_episodes), remaining_rows)
                        sampled_episodes = random.sample(dir_episodes, num_samples)
                        self.episodes.extend(sampled_episodes)
                        remaining_rows -= num_samples
                        used_dirs.add(dir)
                        print(f"Directory {dir}: sampled {num_samples} rows")
                        print(f"Remaining needed: {remaining_rows}")
                        print(f"Unused directories: {len(unused_dirs)}")
                
                if remaining_rows > 0:
                    print(f"Warning: Could not reach target size. Short by {remaining_rows} samples.")
                    print(f"Used {len(used_dirs)} directories out of {len(self.agent_total_dirs)} available.")
        
        # Convert to tensor only after sampling to avoid memory spikes
        if len(self.episodes) > 0:
            self.episodes = torch.stack(self.episodes, dim=0)
        
        print(f"Final dataset size: {len(self.episodes)} rows")
        print(f"Memory usage: {self.episodes.element_size() * self.episodes.nelement() / (1024**3):.2f} GB")

    def filter_black_images(self, batch, threshold=1e-2):
        """Filters out black images from a batch."""
        filtered_images = []
        total_images = len(batch)
        for img in batch:
            if not torch.all(img.abs() <= threshold):
                filtered_images.append(img)

        if filtered_images:
            print(f"Filtered out {total_images - len(filtered_images)}/{total_images} black images")
            return torch.stack(filtered_images)
        else:
            raise ValueError("All images in batch are black")

    def random_crop_img(self, img):
        """Performs random cropping of the input image."""
        max_x = img.shape[2] - self.crop_size
        max_y = img.shape[1] - self.crop_size
        x = random.randint(0, max_x)
        y = random.randint(0, max_y)
        return img[:, y:y + self.crop_size, x:x + self.crop_size]

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, idx):
        return self.episodes[idx]