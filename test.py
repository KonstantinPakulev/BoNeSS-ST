import hydra
import pytorch_lightning as pl

from omegaconf import OmegaConf
from clearml import Task

from source.hydra_config_wrapper import HydraConfigWrapper
from source.test_task import TestTask
from source.data_module import DataModule


@hydra.main(config_path='config', config_name='test_config', version_base='1.1')
def run(config):
    OmegaConf.set_struct(config, False)

    hydra_cfg_wrapper = HydraConfigWrapper(config)

    clearml_task = Task.init(project_name='CVPR 2025',
                             task_name=hydra_cfg_wrapper.test_task_name,
                             task_type='testing',
                             tags=hydra_cfg_wrapper.tags)
    
    hydra_cfg_wrapper.print()
    
    test_task = TestTask.from_config(config, clearml_task, hydra_cfg_wrapper.test_feature_dir_path)
    data_module = DataModule(test_dataset_cfg=config.test.dataset,
                             evaluation_cfg=config.test.evaluation,
                             feature_dir_path=test_task.feature_dir_path)
    
    exp_cfg = config.test.experiment
    
    trainer = pl.Trainer(accelerator=exp_cfg.accelerator,
                         devices=exp_cfg.devices)
    trainer.test(test_task, datamodule=data_module)


if __name__ == "__main__":
    run()
