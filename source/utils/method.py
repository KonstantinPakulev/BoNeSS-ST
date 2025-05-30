import torch
from clearml import InputModel, Task

from source.baselines.hardnet.method import HardNetPS
from source.baselines.sift import SIFT
from source.baselines.superpoint.method import SuperPoint
from source.baselines.r2d2.method import R2D2
from source.baselines.keynet.method import KeyNet
from source.baselines.disk.method import DISK
from source.baselines.rekd.method import REKD
from source.method.syss.base_detectors.shi_tomasi import ShiTomasi
from source.baselines.ness_st.method import NeSSST
from source.method.ness_detector import NeSSDetector


"""
Instantiation utils
"""


def instantiate_method(method_config):
    if method_config.name == 'boness_st':
        method = NeSSDetector.from_config(method_config)

        if 'checkpoint' in method_config:
            ckpt_config = method_config.checkpoint

            task = Task.get_task(task_id=ckpt_config.task_id)

            ckpt = get_checkpoint_from_clearml_task(task, ckpt_config.model_id)
            
            if ckpt is None:
                raise ValueError(f"NeSS model with id {ckpt_config.model_id} not found")

            else:
                print(f"Loaded checkpoint {ckpt_config.model_id} for {method_config.name}")
            
        elif 'checkpoint_url' in method_config:
            input_model = InputModel.import_model(method_config.checkpoint_url, name=method_config.name)

            ckpt = torch.load(input_model.get_weights(), map_location='cpu')
            
            print(f"Loaded checkpoint for {method_config.name}")
        
        else:
            return method
        
        method.load_state_dict({k.replace('ness_detector.', ''): v for k, v in ckpt['state_dict'].items() if not k.startswith('sample_detector.')})

        return method
    
    elif 'checkpoint_url' not in method_config:
        if method_config.name == 'shi_tomasi':
            return ShiTomasi()
        
        elif method_config.name == 'sift':
            return SIFT()
        
        else:
            raise ValueError(f"Handcrafted method {method_config.name} not found")
    
    else:
        input_model = InputModel.import_model(method_config.checkpoint_url, name=method_config.name)
        
        ckpt = torch.load(input_model.get_weights(), map_location='cpu')

        if method_config.name == 'hardnet':
            method = HardNetPS()
            method.load_state_dict(ckpt)
        
        elif method_config.name == 'superpoint':
            method = SuperPoint()
            method.load_state_dict(ckpt)

        elif method_config.name == 'r2d2':
            method = R2D2()
            method.load_state_dict({k.replace('module.', ''): v for k, v in ckpt['state_dict'].items()})

        elif method_config.name == 'keynet':
            method = KeyNet()
            method.load_state_dict(ckpt['state_dict'])

        elif method_config.name == 'disk':
            method = DISK()
            method.load_state_dict(ckpt['extractor'])

        elif method_config.name == 'rekd':
            method = REKD()
            method.load_state_dict(ckpt)

        elif method_config.name == 'ness_st':
            method = NeSSST()
            method.load_state_dict(ckpt)

        else:
            raise ValueError(f"Learned method {method_config.name} not found")
        
        print(f"Loaded checkpoint for {method_config.name}")

        return method


def get_checkpoint_from_clearml_task(task, model_id, return_path=False):
    model_list = task.get_models()['output']

    for model in model_list:
        if model.id == model_id:
            if return_path:
                return model.get_weights()

            else:
                return torch.load(model.get_weights(), map_location='cpu')

    return None


"""
Feature extraction utils
"""


def get_features(detector, descriptor, end_to_end, image, grayscale_image, n):
     if end_to_end is not None:
        return end_to_end.get_features(image, n)

     else:
        if isinstance(detector, NeSSDetector) or \
           isinstance(detector, NeSSST):
            kp = detector.get_keypoints(image, grayscale_image, n)

        elif isinstance(detector, SIFT) or \
             isinstance(detector, SuperPoint) or \
             isinstance(detector, KeyNet) or \
             isinstance(detector, REKD) or \
             isinstance(detector, ShiTomasi):
            kp = detector.get_keypoints(grayscale_image, n)
        
        else:
            kp = detector.get_keypoints(image, n)
        
        if isinstance(descriptor, HardNetPS):
            kp_desc = descriptor.get_descriptors(grayscale_image, kp)
            
        else:
            kp_desc = descriptor.get_descriptors(image, kp)
        
        return kp, kp_desc
