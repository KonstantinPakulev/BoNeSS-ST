import torch
import pytorch_lightning as pl
import torchvision.transforms as transforms

from omegaconf import DictConfig
from typing import Iterator, Sized, Optional
from torch.utils.data import DataLoader, RandomSampler, Sampler

from source.datasets.image_dataset import ImageDataset
from source.datasets.image_pairs_dataset import ImagePairsDataset
from source.datasets.feature_dataset import FeatureDataset, feature_dataset_collate_fn


class DataModule(pl.LightningDataModule):

    def __init__(self, 
                 train_dataset_cfg=None, 
                 val_dataset_cfg=None, 
                 test_dataset_cfg=None, 
                 predict_dataset_cfg=None,
                 evaluation_cfg=None,
                 feature_dir_path=None):
        super().__init__()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None

        self.train_dataset_cfg = train_dataset_cfg
        self.val_dataset_cfg = val_dataset_cfg
        self.test_dataset_cfg = test_dataset_cfg
        self.predict_dataset_cfg = predict_dataset_cfg

        self.evaluation_cfg = evaluation_cfg

        self.feature_dir_path = feature_dir_path
    
    def setup(self, stage=None):
        if stage == 'fit':
            self.train_dataset = get_training_dataset(self.train_dataset_cfg)
                        
            self.val_dataset = get_testing_dataset(self.val_dataset_cfg, self.evaluation_cfg, self.feature_dir_path)
            
        elif stage == 'test':
            self.test_dataset = get_testing_dataset(self.test_dataset_cfg, self.evaluation_cfg, self.feature_dir_path)
                        
        elif stage == 'predict':
            if 'pairs_csv_rel_path' in self.predict_dataset_cfg:
                self.predict_dataset = ImagePairsDataset(self.predict_dataset_cfg.root_path, 
                                                         self.predict_dataset_cfg.pairs_csv_rel_path, 
                                                         transforms.Compose([transforms.ToPILImage(),
                                                                             transforms.ToTensor(),
                                                                             CropToMultipleOf16()]))
            else:
                random_crop_cfg = self.predict_dataset_cfg.transforms.random_crop
                
                self.predict_dataset = ImageDataset(self.predict_dataset_cfg.root_path, 
                                                    self.predict_dataset_cfg.csv_rel_path, 
                                                    transforms.Compose([transforms.ToPILImage(),
                                                                        transforms.CenterCrop((random_crop_cfg.height, random_crop_cfg.width)),
                                                                        transforms.ToTensor(),
                                                                        CropToMultipleOf16()]))
        
    def train_dataloader(self):
        return get_training_dataloader(self.train_dataset, self.train_dataset_cfg)
    
    def val_dataloader(self):
        return get_testing_dataloader(self.val_dataset, self.val_dataset_cfg, self.evaluation_cfg)
    
    def test_dataloader(self):
        return get_testing_dataloader(self.test_dataset, self.test_dataset_cfg, self.evaluation_cfg)
    
    def predict_dataloader(self):
        return DataLoader(self.predict_dataset, batch_size=1)


"""
Transforms
"""

class CropToMultipleOf16:

    def __call__(self, image: torch.Tensor):
        crop_rect = [0, 0, image.size()[1], image.size()[2]]

        for i, size in enumerate(image.size()[1:]):
            if size % 16 != 0:
                new_size = (size // 16) * 16
                offset = int(round((size - new_size) / 2.))
            else:
                new_size = size
                offset = 0
            
            crop_rect[i] = offset
            crop_rect[i + 2] = new_size
        
        return transforms.functional.crop(image, crop_rect[0], crop_rect[1], crop_rect[2], crop_rect[3])

"""
Dataset utils
"""

def get_training_dataset(dataset_cfg: DictConfig):
    random_crop_cfg = dataset_cfg.transforms.random_crop

    train_transform = transforms.Compose([transforms.ToPILImage(),
                                          transforms.CenterCrop((random_crop_cfg.height, random_crop_cfg.width)),
                                          transforms.ToTensor(),
                                          CropToMultipleOf16()])
                                                    
    return ImageDataset(dataset_cfg.root_path, 
                        dataset_cfg.csv_rel_path, 
                        train_transform)

def get_testing_dataset(dataset_cfg: DictConfig, evaluation_cfg: DictConfig, feature_dir_path):
    if evaluation_cfg.only_features:
        return ImageDataset(dataset_cfg.root_path, 
                            dataset_cfg.csv_rel_path, 
                            transforms.Compose([transforms.ToPILImage(),
                                                transforms.ToTensor(),
                                                CropToMultipleOf16()]))
    
    elif evaluation_cfg.only_task:
        return FeatureDataset(dataset_cfg.root_path, 
                              dataset_cfg.pairs_csv_rel_path,
                              feature_dir_path)
            
    else:
        image_dataset = ImageDataset(dataset_cfg.root_path, 
                            dataset_cfg.csv_rel_path, 
                            transforms.Compose([transforms.ToPILImage(),
                                                transforms.ToTensor(),
                                                CropToMultipleOf16()]))
    
        feature_dataset = FeatureDataset(dataset_cfg.root_path, 
                                        dataset_cfg.pairs_csv_rel_path,
                                        feature_dir_path)

        return [image_dataset, feature_dataset]


"""
Sampler
"""

class SequentialSampler(Sampler[int]):

    def __init__(self, data_source: Sized, num_samples: Optional[int] = None) -> None:
        self.data_source = data_source
        self.num_samples = num_samples
    
    def __iter__(self) -> Iterator[int]:
        if self.num_samples is None:
            return iter(range(len(self.data_source)))
        
        else:
            return iter(range(self.num_samples))

    def __len__(self) -> int:
        if self.num_samples is None:
            return len(self.data_source)
        
        else:
            return self.num_samples



"""
Dataloader utils
"""

def get_training_dataloader(dataset: ImageDataset, dataset_cfg: DictConfig):
    loader_cfg = dataset_cfg.loader

    if loader_cfg.shuffle:
        sampler = RandomSampler(dataset, num_samples=loader_cfg.num_samples, replacement=True)
    else:
        sampler = SequentialSampler(dataset)

    return DataLoader(dataset,
                      sampler=sampler,
                      batch_size=loader_cfg.batch_size,
                      num_workers=loader_cfg.num_workers)


def get_testing_dataloader(dataset, dataset_cfg: DictConfig, evaluation_cfg: DictConfig):
    loader_cfg = dataset_cfg.loader
    pairs_loader_cfg = dataset_cfg.pairs_loader

    if evaluation_cfg.only_features:
        return DataLoader(dataset,
                          sampler=SequentialSampler(dataset, num_samples=loader_cfg.get('num_samples', None)),
                          batch_size=1,
                          num_workers=loader_cfg.num_workers)
    
    elif evaluation_cfg.only_task:
        return DataLoader(dataset,
                          sampler=SequentialSampler(dataset, num_samples=pairs_loader_cfg.get('num_samples', None)),
                          batch_size=pairs_loader_cfg.batch_size,
                          num_workers=pairs_loader_cfg.num_workers,
                          collate_fn=feature_dataset_collate_fn)
    
    else:
        dataloader = DataLoader(dataset[0],
                                sampler=SequentialSampler(dataset[0], num_samples=loader_cfg.get('num_samples', None)),
                                batch_size=1,
                                num_workers=loader_cfg.num_workers)
        
        pairs_dataloader = DataLoader(dataset[1],  
                                      sampler=SequentialSampler(dataset[1], num_samples=pairs_loader_cfg.get('num_samples', None)),
                                      batch_size=pairs_loader_cfg.batch_size,
                                      num_workers=pairs_loader_cfg.num_workers,
                                      collate_fn=feature_dataset_collate_fn)
        
        return [dataloader, pairs_dataloader]
