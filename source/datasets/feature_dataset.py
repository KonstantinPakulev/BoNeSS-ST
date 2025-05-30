import numpy as np
import pandas as pd
import h5py
import torch

from pathlib import Path
from torch.utils.data import Dataset

from source.utils.io import load_homography, load_h5py_calibration, load_npy_calibration, load_h5py_intrinsics, load_h5py_rel_pose, load_txt_calibration, get_inference_filename, load_features


class FeatureDataset(Dataset):

    def __init__(self, root_path, csv_rel_path, feature_dir_path):
        self.root_path = Path(root_path)
        self.df = pd.read_csv(self.root_path / csv_rel_path, index_col=[0])

        self.feature_dir_path = feature_dir_path

    def __getitem__(self, index):
        df_row = self.df.iloc[index]

        item = {'index': index}

        for i in range(1, 3):
            if f'calib{i}' in df_row:
                calib_path = self.root_path / df_row[f'calib{i}']

                if calib_path.suffix == '.h5':
                    extrinsics, intrinsics = load_h5py_calibration(calib_path)

                elif calib_path.suffix == '.npy':
                    extrinsics, intrinsics = load_npy_calibration(calib_path)
                
                elif calib_path.suffix == '.txt':
                    extrinsics, intrinsics = load_txt_calibration(calib_path)

                else:
                    raise ValueError(f"Expected .h5 or .npy file extension for calibration file, got {calib_path.suffix}")

                item[f'extrinsics{i}'] = extrinsics
                item[f'intrinsics{i}'] = intrinsics

            if f'intrinsics{i}' in df_row:
                item[f'intrinsics{i}'] = load_h5py_intrinsics(self.root_path / df_row[f'intrinsics{i}'])
            
            image_rel_path = Path(df_row[f'image{i}'])
            feature_path = self.feature_dir_path / get_inference_filename(image_rel_path.stem, str(image_rel_path.parent))

            item[f'image_rel_path{i}'] = str(image_rel_path)

            kp, kp_desc, offset, image_dims = load_features(feature_path)

            item[f'kp{i}'] = kp
            item[f'kp_desc{i}'] = kp_desc
            item[f'offset{i}'] = offset

            if image_dims is not None:
                item[f'image_dims{i}'] = image_dims
        
        if 'rel_pose' in df_row:
            item['rel_pose'] = load_h5py_rel_pose(self.root_path / df_row['rel_pose'])
        
        if 'h1' in df_row:
            h_path = self.root_path / df_row['h1']
            item['h'] = load_homography(h_path)
        
        return item
    
    def __len__(self):
        return len(self.df)
    

"""
Collate function
"""


def feature_dataset_collate_fn(batch):
    max_num_kp1 = max(len(item['kp1']) for item in batch)
    max_num_kp2 = max(len(item['kp2']) for item in batch)

    new_batch = []
    for item in batch:
        new_item = item.copy()

        for i in range(1, 3):
            kp = item[f'kp{i}']
            offset = item[f'offset{i}']
            kp_desc = item[f'kp_desc{i}']
            max_num_kp = max_num_kp1 if i == 1 else max_num_kp2

            if len(kp) < max_num_kp:
                pad_size = max_num_kp - len(kp)
                new_item[f'kp{i}'] = np.pad(kp, ((0, pad_size), (0, 0)), mode='constant')
                new_item[f'kp_desc{i}'] = np.pad(kp_desc, ((0, pad_size), (0, 0)), mode='constant')
                new_item[f'kp_mask{i}'] = np.pad(np.ones(len(kp)), (0, pad_size), mode='constant').astype(bool)

            else:
                new_item[f'kp_mask{i}'] = np.ones(len(kp), dtype=bool)
            
            new_item[f'offset{i}'] = offset
        
        new_batch.append(new_item)
    
    return torch.utils.data.dataloader.default_collate(new_batch)
