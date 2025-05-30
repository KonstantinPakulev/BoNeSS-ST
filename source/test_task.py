import shutil
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import pytorch_lightning as pl

from omegaconf import DictConfig
from pathlib import Path
from clearml import Task

from source.baselines.disk.method import DISK
from source.utils.method import instantiate_method
from source.evaluation.two_view_geometry import match_and_estimate_rel_pose_error_batch, relative_pose_accuracy
from source.evaluation.classical_metrics import repeatability, mean_matching_accuracy
from source.utils.io import save_features, get_inference_filename
from source.utils.method import get_features


class TestTask(pl.LightningModule):
          
     @staticmethod
     def from_config(cfg: DictConfig, 
                     clearml_task: Task, 
                     feature_dir_path):
          if cfg.test.evaluation.only_task:
               detector = None
               descriptor = None
               end_to_end = None
               
          else:
               if 'end_to_end' in cfg.method:
                    detector = None
                    descriptor = None
                    end_to_end = instantiate_method(cfg.method.end_to_end)

               else:
                    detector = instantiate_method(cfg.method.detector)
                    descriptor = instantiate_method(cfg.method.descriptor)
                    end_to_end = None
          
          return TestTask(cfg.test.evaluation, 
                          clearml_task,
                          feature_dir_path,
                          detector, descriptor, 
                          end_to_end)
     
     def __init__(self, 
                  evaluation_cfg: DictConfig, 
                  clearml_task: Task,
                  feature_dir_path,
                  detector: nn.Module=None, descriptor: nn.Module=None,
                  end_to_end: DISK=None):
          super().__init__()

          self.detector = detector
          self.descriptor = descriptor
          self.end_to_end = end_to_end

          self.evaluation_cfg = evaluation_cfg
          self.clearml_task = clearml_task
          self.feature_dir_path = feature_dir_path
          self.test_step_outputs = []
          
          self.prev_dataloader_idx = -1

     def on_test_epoch_start(self) -> None:
          self.feature_dir_path.mkdir(exist_ok=True, parents=True)

     def test_step(self, batch, batch_idx, dataloader_idx=0):
          if self.evaluation_cfg.only_features:
               return self._calculate_and_save_features(batch, batch_idx)
          
          elif self.evaluation_cfg.only_task:
               return self.test_step_outputs.append(self._execute_evaluation_task(batch, batch_idx))
          
          else:
               if self.prev_dataloader_idx != dataloader_idx:
                    self.prev_dataloader_idx = dataloader_idx
                    
                    if dataloader_idx == 1:
                         del self.detector
                         self.detector = None

                         del self.descriptor
                         self.descriptor = None

                         torch.cuda.empty_cache()
               
               if dataloader_idx == 0:
                    return self._calculate_and_save_features(batch, batch_idx)
               
               else:
                    return self.test_step_outputs.append(self._execute_evaluation_task(batch, batch_idx))
          
     def _calculate_and_save_features(self, batch, batch_idx):
          image, grayscale_image = batch['image'], batch['grayscale_image']
          img_filename, img_dir_rel_path = batch['image_filename'][0], batch['image_dir_rel_path'][0]
          offset = batch['offset']

          kp, kp_desc = get_features(self.detector, self.descriptor, self.end_to_end, image, grayscale_image, self.evaluation_cfg.n)

          inference_filepath = self.feature_dir_path / get_inference_filename(img_filename, img_dir_rel_path)

          if self.evaluation_cfg.task == 'two_view_geometry':
               save_features(inference_filepath, kp.squeeze(0), kp_desc.squeeze(0), offset.squeeze(0))

          elif self.evaluation_cfg.task == 'classical_metrics':
               save_features(inference_filepath, kp.squeeze(0), kp_desc.squeeze(0), offset.squeeze(0), torch.tensor(image.shape[2:]))

          else:
               raise ValueError(f'Unknown evaluation task: {self.evaluation_cfg.task}')
          

     def _execute_evaluation_task(self, batch, batch_idx):
          if self.evaluation_cfg.task == 'two_view_geometry':
               results = match_and_estimate_rel_pose_error_batch(batch, 
                                                                 self.evaluation_cfg.ratio_test_thr, 
                                                                 self.evaluation_cfg.parallelize, 
                                                                 self.evaluation_cfg.estimator)
               
               return np.concatenate([batch['index'].cpu().numpy().reshape(-1, 1), 
                                        np.array(batch['image_rel_path1']).reshape(-1, 1),
                                        np.array(batch['image_rel_path2']).reshape(-1, 1),
                                        results], 
                                        axis=1)
          
          elif self.evaluation_cfg.task == 'classical_metrics':
               offset1, offset2 = batch['offset1'].unsqueeze(1), batch['offset2'].unsqueeze(1)
               image_dims1, image_dims2 = batch['image_dims1'].unsqueeze(1), batch['image_dims2'].unsqueeze(1)

               rep = repeatability(batch['kp1'] + offset1, 
                                   batch['kp2'] + offset2,
                                   offset1, offset2,
                                   batch['kp_mask1'], batch['kp_mask2'],
                                   batch['h'],
                                   image_dims1, image_dims2,
                                   self.evaluation_cfg.max_px_err_thr)
               
               mma = mean_matching_accuracy(batch['kp1'] + offset1, 
                                            batch['kp2'] + offset2, 
                                            batch['kp_desc1'], batch['kp_desc2'], 
                                            batch['kp_mask1'], batch['kp_mask2'],
                                            batch['h'],
                                            self.evaluation_cfg.max_px_err_thr)
               
               results = np.concatenate([batch['index'].cpu().numpy().reshape(-1, 1),
                                         np.array(batch['image_rel_path1']).reshape(-1, 1),
                                         np.array(batch['image_rel_path2']).reshape(-1, 1),
                                         rep,
                                         mma], 
                                         axis=1)

               return results
          
          else:
               raise ValueError(f'Unknown evaluation task: {self.evaluation_cfg.task}')
          
     def on_test_epoch_end(self):
          if self.evaluation_cfg.clean_up_features:
               shutil.rmtree(self.feature_dir_path)

          if self.evaluation_cfg.only_features:
               return
          
          elif self.evaluation_cfg.only_task:
               eval_task_output = np.concatenate(self.test_step_outputs, axis=0)

          else:
               eval_task_output = np.concatenate(self.test_step_outputs, axis=0)

          self.test_step_outputs.clear()

          if self.evaluation_cfg.task == 'two_view_geometry':
               columns = ['index', 'image_rel_path1', 'image_rel_path2', 'R_err', 't_err', 'success', 'num_inliers']
               
               r_err = eval_task_output[:, 3].astype(np.float32)
               t_err = eval_task_output[:, 4].astype(np.float32)
               num_inliers = eval_task_output[:, 6].astype(np.float32)
               
               r_mAA = np.mean(relative_pose_accuracy(r_err, self.evaluation_cfg.max_r_err_thr), axis=-1)
               t_mAA = np.mean(relative_pose_accuracy(t_err, self.evaluation_cfg.max_t_err_thr), axis=-1)
               mean_num_inliers = np.mean(num_inliers)

               self.log('r_mAA', r_mAA)
               self.log('t_mAA', t_mAA)
               self.log('mean_num_inliers', mean_num_inliers)
          
          elif self.evaluation_cfg.task == 'classical_metrics':
               columns = ['index', 'image_rel_path1', 'image_rel_path2'] + \
               [f'rep_{i}' for i in range(self.evaluation_cfg.max_px_err_thr)] + \
               [f'mma_{i}' for i in range(self.evaluation_cfg.max_px_err_thr)]

          else:
               raise ValueError(f'Unknown evaluation task: {self.evaluation_cfg.task}')
                    
          df_path = Path.cwd() / 'eval_task_output.csv'

          eval_task_output_df = pd.DataFrame(eval_task_output, columns=columns)
          eval_task_output_df.to_csv(df_path, index=False)

          self.clearml_task.upload_artifact(name='eval_task_output.csv', artifact_object=df_path)
