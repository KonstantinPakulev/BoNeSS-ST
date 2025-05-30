from source.datasets.image_dataset import ImageDataset

from source.utils.common_utils import get_offset
from source.utils.io import load_image, load_h5py_calibration, load_homography, load_h5py_rel_pose, load_h5py_intrinsics, load_npy_calibration, load_txt_calibration


class ImagePairsDataset(ImageDataset):
    
    def __getitem__(self, index):
        df_row = self.df.iloc[index]
                
        item = {'index': index}
        
        for i in range(1, 3):
            image_path = self.root_path / df_row[f'image{i}']
            image, grayscale_image = load_image(image_path)
            
            item[f'filename{i}'] = image_path.stem
            item[f'image{i}'] = image
            item[f'grayscale_image{i}'] = grayscale_image

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
            
            if self.image_transform is not None:
                old_image = item[f'image{i}']

                item[f'image{i}'] = self.image_transform(old_image)
                item[f'grayscale_image{i}'] = self.image_transform(item[f'grayscale_image{i}'])

                item[f'offset{i}'] = get_offset(old_image, item[f'image{i}'])

        if 'rel_pose' in df_row:
            item['rel_pose'] = load_h5py_rel_pose(self.root_path / df_row['rel_pose'])

        if 'h1' in df_row:
            h_path = self.root_path / df_row['h1']
            item['h'] = load_homography(h_path)

        return item
