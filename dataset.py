import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
from torchvision import transforms
from external.YOLOX.yolox.data import ValTransform
import glob
from natsort import natsorted

def get_mot_loader(data_dir, workers=4, size=(640,640)):

        data_root = os.path.join(data_dir)
        
        class CustomDataset(Dataset):
            def __init__(self, root_dir, size):
                self.root_dir = root_dir
                self.size = size
                self.img_files = natsorted(glob.glob(os.path.join(root_dir, "*.jpg")))
                
            def __len__(self):
                return len(self.img_files)
                
            def __getitem__(self, idx):
                img_path = self.img_files[idx]
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Original image for visualization
                np_img = img.copy()
                
                # Resize if needed
                if self.size != (img.shape[0], img.shape[1]):
                    img = cv2.resize(img, (self.size[1], self.size[0]))
                    np_img = cv2.resize(np_img, (self.size[1], self.size[0]))
                
                # Convert to float and normalize
                img = img.astype(np.float32) / 255.0
                img = torch.from_numpy(img).permute(2, 0, 1)
                
                # Create empty label and info (no annotations)
                label = torch.zeros((0, 6))  # empty label tensor
                info = {
                    'file_name': os.path.basename(img_path),
                    'id': idx,
                    'frame_id': int(os.path.basename(img_path).split('.')[0]),
                    'video_id': 0,
                    'height': img.shape[1],
                    'width': img.shape[2],
                    'file_path': os.path.join('cam0', 'img1', os.path.basename(img_path))  # MOT 형식의 경로
                }
                
                # Convert info to list format as expected by main.py
                info_list = [
                    info['id'],           # idx 0: image id
                    info['video_id'],     # idx 1: video id
                    info['frame_id'],     # idx 2: frame id
                    info['file_name'],    # idx 3: file name
                    info['file_path'],    # idx 4: file path
                    info['height'],       # idx 5: height
                    info['width']         # idx 6: width
                ]
                
                return (img, np_img), label, info_list, idx
        
        dataset = CustomDataset(data_root, size)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, 
                              num_workers=workers, pin_memory=True)
        return dataloader
        
