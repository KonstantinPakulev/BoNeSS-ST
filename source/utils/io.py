import numpy as np
from skimage import io
from skimage.color import rgb2gray
import torch
import h5py

from pathlib import Path


def load_image(image_path):
    image = io.imread(image_path)

    if len(image.shape) == 2:
        grayscale_image = image
        image = np.stack([image] * 3, axis=-1)

    else:
        grayscale_image = rgb2gray(image).astype(np.float32)

    return image, grayscale_image


def load_h5py_calibration(calib_path):
    with h5py.File(calib_path, 'r') as file:
        extrinsics = np.zeros((4, 4))
        extrinsics[:3, :3] =  np.array(file['R'])
        extrinsics[:3, 3] = np.array(file['T'])
        extrinsics[3, 3] = 1

        intrinsics = np.array(file['K'])

    return extrinsics, intrinsics


def load_txt_calibration(extrinsics_path):
    with open(extrinsics_path) as f:
        extrinsics = [i.strip().split(' ') for i in f.readlines()]
        extrinsics = np.linalg.inv(np.array(extrinsics, dtype=np.float32))
    
    intrinsics_path = Path(extrinsics_path).parent.parent / 'intrinsic' / 'intrinsic_color.txt'

    with open(intrinsics_path) as f:
        intrinsics = [i.strip().split(' ') for i in f.readlines()]
        intrinsics = np.array(intrinsics, dtype=np.float32)[:3, :3]
    
    return extrinsics, intrinsics


def load_npy_calibration(calib_path):
    calib_data = np.load(calib_path, allow_pickle=True).item()

    return calib_data['extrinsics'], calib_data['intrinsics']


def load_h5py_rel_pose(rel_pose_path):
    with h5py.File(rel_pose_path, 'r') as file:
        rel_pose = np.array(file['T12'])

    return rel_pose


def load_h5py_intrinsics(intrinsics_path):
    with h5py.File(intrinsics_path, 'r') as file:
        intrinsics = np.array(file['K'])

    return intrinsics


def load_homography(homography_path):
    return np.array(np.asmatrix(np.loadtxt(homography_path, dtype=np.float32)))


def get_inference_filename(image_filename: str, image_dir_rel_path: str):
    return image_dir_rel_path.replace('/', '_') + '_' + f"{image_filename}.h5py"


def save_features(inference_filepath: Path, 
                  kp: torch.Tensor, kp_desc: torch.Tensor, 
                  offset: torch.Tensor, image_dims: torch.Tensor=None):
    with h5py.File(inference_filepath, 'w') as file:
        file.create_dataset('kp', data=kp.flip([-1]).cpu().numpy())
        file.create_dataset('kp_desc', data=kp_desc.cpu().numpy())

        file.create_dataset('offset', data=offset.flip([-1]).cpu().numpy())
        
        if image_dims is not None:
            file.create_dataset('image_dims', data=image_dims.flip([-1]).cpu().numpy())


def load_features(feature_path):
    with h5py.File(feature_path, 'r') as file:
        kp = np.array(file['kp'])
        kp_desc = np.array(file['kp_desc'])
        offset = np.array(file['offset'])
        image_dims = None
        
        if 'image_dims' in file:
            image_dims = np.array(file['image_dims'])
            
        return kp, kp_desc, offset, image_dims


def save_figure(fig, base_save_dir, file_name, rel_path=None, prefix=None, suffix=None):
    save_dir_path = base_save_dir / rel_path if rel_path is not None else base_save_dir
    
    if prefix is not None:
        file_name = f"{prefix}_{file_name}"

    if suffix is not None:
        file_name = f"{file_name}_{suffix}"
            
    save_dir_path.mkdir(parents=True, exist_ok=True)

    fig.savefig(save_dir_path / f"{file_name}.pdf", bbox_inches='tight')