import hydra
import pytorch_lightning as pl

from omegaconf import OmegaConf
from clearml import Task
from pytorch_lightning.callbacks import ModelCheckpoint

from source.hydra_config_wrapper import HydraConfigWrapper
from source.train_task import TrainTask
from source.data_module import DataModule
from source.utils.method import get_checkpoint_from_clearml_task


@hydra.main(config_path='config', config_name='train_config', version_base='1.1')
def run(config):
    OmegaConf.set_struct(config, False)

    hydra_cfg_wrapper = HydraConfigWrapper(config)

    exp_cfg = config.train.experiment

    task_id = exp_cfg.get('task_id', None)
    ckpt_path = None

    # TODO: can we move this to checkpoint and instead just have an experiment flag?
    if task_id is not None:
        clearml_task = Task.init(project_name='CVPR 2025',
                                 task_name=hydra_cfg_wrapper.train_task_name,
                                 task_type='training',
                                 continue_last_task=task_id)
        
        ckpt_path = get_checkpoint_from_clearml_task(clearml_task, exp_cfg.model_id, return_path=True)

    else:
        clearml_task = Task.init(project_name='CVPR 2025',
                                 task_name=hydra_cfg_wrapper.train_task_name,
                                 task_type='training',
                                 tags=hydra_cfg_wrapper.tags)
        
    hydra_cfg_wrapper.print()
        
    train_task = TrainTask.from_config(config, hydra_cfg_wrapper.val_feature_dir_path)
    data_module = DataModule(train_dataset_cfg=config.train.dataset,
                             val_dataset_cfg=config.val.dataset,
                             evaluation_cfg=config.val.evaluation,
                             feature_dir_path=train_task.feature_dir_path)
    
    model_checkpoint = ModelCheckpoint(dirpath=hydra.core.hydra_config.HydraConfig.get().runtime.output_dir,
                                       filename='epoch={epoch:02d}-avg_mAA={avg_mAA:.4f}',
                                       save_top_k=exp_cfg.model_checkpoint.save_top_k,
                                       monitor='avg_mAA',
                                       mode='max',
                                       auto_insert_metric_name=False,
                                       save_last=True)

    trainer = pl.Trainer(accelerator=exp_cfg.accelerator,
                         devices=exp_cfg.devices,
                         max_epochs=exp_cfg.max_epochs,
                         log_every_n_steps=exp_cfg.log_every_n_steps,
                         num_sanity_val_steps=0,
                         callbacks=[model_checkpoint])
    
    trainer.fit(train_task, datamodule=data_module, ckpt_path=ckpt_path)


if __name__ == "__main__":
    run()
