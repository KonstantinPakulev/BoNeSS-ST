from omegaconf import OmegaConf
from pathlib import Path

from hydra.core.hydra_config import HydraConfig


class HydraConfigWrapper:

    def __init__(self, config):
        self.config = config
        
        if self.is_in_mock_mode():
            if self.is_in_train_mode():
                self.config.train.experiment.max_epochs = 2
                self.config.train.experiment.log_every_n_steps = 1
                self.config.train.dataset.loader.num_samples = self.config.train.dataset.loader.batch_size * 2

                self.config.val.dataset.loader.num_samples = 8
                self.config.val.dataset.pairs_loader.batch_size = 2
                self.config.val.dataset.pairs_loader.num_samples = 4
            
            if self.is_in_test_mode():
                self.config.test.dataset.loader.num_samples = 8
                self.config.test.dataset.pairs_loader.batch_size = 2
                self.config.test.dataset.pairs_loader.num_samples = 4

    def is_in_mock_mode(self):
        return self.config.get('mock', False)
    
    def is_in_train_mode(self):
        return 'train' in self.config
    
    def is_in_test_mode(self):
        return 'test' in self.config
    
    def print(self):
        print("\n")
        print("Configuration file:")
        print("-" * 90)
        print(OmegaConf.to_yaml(self.config))
        print("-" * 90)
    
    @property
    def train_task_name(self):
        hydra_task_list = HydraConfig.get().overrides.task

        task_name_part_list = []

        for hydra_task in hydra_task_list:
            if '+methods/detector' in hydra_task:
                task_name_part_list.append(hydra_task.replace('+methods/', ''))
            
            elif '+train/criterion' in hydra_task:
                task_name_part_list.append(hydra_task.replace('+train/', '').replace('setup/', ''))

            elif '/' not in hydra_task:
                task_name_part_list.append(hydra_task.replace('train.', ''))
            
        return ','.join(task_name_part_list)
    
    @property
    def test_task_name(self):
        if 'task_name' in self.config.test.experiment:
            if self.is_in_mock_mode():
                return self.config.test.experiment.task_name + ',mock=True'
                        
            else:
                return self.config.test.experiment.task_name
                        
        else:
            hydra_task_list = HydraConfig.get().overrides.task

            task_name_part_list = []

            for hydra_task in hydra_task_list:
                if '+method/detector' in hydra_task or \
                    '+method/descriptor' in hydra_task or \
                    '+method/end_to_end' in hydra_task:
                    task_name_part_list.append(hydra_task.replace('+method/', ''))
                
                elif '/' not in hydra_task.split('=')[0]:
                    task_name_part_list.append(hydra_task.replace('test.', '').replace('+', '').replace('"', ''))
                
            return ','.join(task_name_part_list)
    
    @property
    def tags(self):
        if self.is_in_test_mode():
            tags = self.config.test.experiment.get('tags', [])
            
            if self.config.test.dataset.name not in tags:
                tags = [self.config.test.dataset.name] + tags
        else:
            tags = self.config.train.experiment.get('tags', [])

        if self.is_in_mock_mode():
            tags.append('mock')
        
        return tags

    @property   
    def val_feature_dir_path(self):
        return Path(self.config.val.evaluation.feature_dir_path) / 'val'
    
    @property
    def test_feature_dir_path(self):
        if 'end_to_end' in self.config.method:
            method_name = self.config.method.end_to_end.name
        else:
            if self.config.method.detector.name in ['ness_st', 'boness_st'] and \
               'checkpoint_url' not in self.config.method.detector:
                method_name = self.config.method.detector.name + '@' + \
                              self.config.method.detector.checkpoint.model_id + '_' + \
                              self.config.method.descriptor.name
                
            else:
                method_name = self.config.method.detector.name + '_' + \
                              self.config.method.descriptor.name
        
          
        test_feature_dir_path = Path(self.config.test.evaluation.feature_dir_path) / \
                                "test" / \
                                self.config.test.dataset.name / \
                                self.config.test.evaluation.task
        
        if self.config.test.evaluation.task == 'two_view_geometry':
            test_feature_dir_path = test_feature_dir_path / \
                                    self.config.test.evaluation.estimator.name / \
                                    method_name
        
        if self.is_in_mock_mode():
            return test_feature_dir_path / 'mock'
        
        else:
            return test_feature_dir_path
        
