import torch
import numpy as np
import pytorch_lightning as pl
import shutil

from torch.optim import Adam
from omegaconf import DictConfig

from source.method.syss.patch_detector import PatchDetector
from source.method.syss.synthetic_homography_distribution import SyntheticHomographyDistribution
from source.method.syss.beta_eme import BetaEME
from source.method.syss.warped_patch_generator import WarpedPatchGenerator
from source.method.syss.base_detectors.instantiate_base_detector import instantiate_base_detector

from source.method.ness_detector import NeSSDetector
from source.utils.method import instantiate_method
from source.evaluation.two_view_geometry import match_and_estimate_rel_pose_error_batch
from source.evaluation.two_view_geometry import relative_pose_accuracy
from source.utils.io import get_inference_filename, save_features


class TrainTask(pl.LightningModule):

    @staticmethod
    def from_config(cfg: DictConfig, 
                    feature_dir_path):
        ness_detector = instantiate_method(cfg.method.detector)

        sample_detector = NeSSDetector(ness_detector.neur_beta_eme_model,
                                       instantiate_base_detector(cfg.train.criterion.sample_detector.base_detector))
                
        patch_detector = PatchDetector(instantiate_base_detector(cfg.train.criterion.patch_detector.base_detector),
                                       cfg.train.criterion.patch_detector.unreliable_measurement_type)
                
        synth_h_dist = SyntheticHomographyDistribution(cfg.train.criterion.homography_distribution.beta,
                                                       patch_detector.patch_size,
                                                       cfg.train.criterion.homography_distribution.sampling_strategy)
        
        beta_eme = BetaEME(cfg.train.criterion.beta_eme.bound,
                           cfg.train.criterion.beta_eme.use_decomposition)
        
        return TrainTask(ness_detector, cfg.method.descriptor,
                         sample_detector, patch_detector, synth_h_dist, beta_eme,
                         cfg.train.criterion.sample_detector.salient_thresh,
                         cfg.train.criterion.sample_detector.noise_thresh,
                         cfg.train.criterion.n, cfg.train.criterion.m,
                         cfg.train.optimizer.lr, 
                         cfg.val.evaluation,
                         feature_dir_path)
        
    def __init__(self, 
                 ness_detector: NeSSDetector, descriptor_cfg: DictConfig,
                 sample_detector: NeSSDetector,
                 patch_detector: PatchDetector,
                 synth_h_dist: SyntheticHomographyDistribution,
                 beta_eme: BetaEME,
                 salient_thresh: float, noise_thresh: float,
                 n: int, m: int, lr: float,
                 evaluation_cfg: DictConfig,
                 feature_dir_path):
        super().__init__()

        self.ness_detector = ness_detector
        self.descriptor = None
        self.descriptor_cfg = descriptor_cfg

        self.sample_detector = sample_detector
        self.patch_detector = patch_detector
        self.synth_h_dist = synth_h_dist
        self.beta_eme = beta_eme

        self.salient_thresh = salient_thresh
        self.noise_thresh = noise_thresh

        self.n = n
        self.m = m

        self.lr = lr

        self.evaluation_cfg = evaluation_cfg

        self.feature_dir_path = feature_dir_path
        self.validation_step_outputs = []

    def training_step(self, batch, batch_idx):
        image, grayscale_image = batch['image'], batch['grayscale_image']

        kp_samples = self.sample_detector.get_keypoint_samples(image, grayscale_image,
                                                               self.n,
                                                               self.synth_h_dist.border_size,
                                                               self.salient_thresh, self.noise_thresh)
                
        h = self.synth_h_dist.sample(self.m, image.device)

        w_patch_gen = WarpedPatchGenerator(h, self.synth_h_dist.patch_scaled_size, image.device)
        kp_proj_gen = w_patch_gen.get_keypoint_projection_generator(grayscale_image, kp_samples.kp, self.patch_detector.
                                                                    patch_size)
        
        kp_proj = kp_proj_gen.generate_keypoint_projections(self.patch_detector)

        kp_synth_beta_eme = self.beta_eme.calculate(kp_proj)
        kp_synth_beta_eme = self.beta_eme.replace_with_max_value(kp_synth_beta_eme, kp_samples.cand_kp_noise_mask,
                                                                 self.patch_detector, self.synth_h_dist.beta)

        loss = 0.5 * (kp_samples.kp_neur_beta_eme - kp_synth_beta_eme) ** 2
        loss = kp_mask_loss(loss, kp_samples.cand_kp_mask)

        self.log('loss', loss)

        return loss

    def on_validation_epoch_start(self):
        self.descriptor = instantiate_method(self.descriptor_cfg)
        self.descriptor.to(self.device)

        self.feature_dir_path.mkdir(exist_ok=True, parents=True)
    
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        if dataloader_idx == 0:
            return self._validation_step_features(batch, batch_idx)
        
        elif dataloader_idx == 1:
            results = match_and_estimate_rel_pose_error_batch(batch, 
                                                              self.evaluation_cfg.ratio_test_thr, 
                                                              self.evaluation_cfg.parallelize, 
                                                              self.evaluation_cfg.estimator)

            self.validation_step_outputs.append(results[:, :2])
    
    def _validation_step_features(self, batch, batch_idx):
        image, grayscale_image = batch['image'], batch['grayscale_image']
        img_filename, img_dir_rel_path = batch['image_filename'][0], batch['image_dir_rel_path'][0]
        offset = batch['offset']

        kp = self.ness_detector.get_keypoints(image, grayscale_image, self.evaluation_cfg.n)
        kp_desc = self.descriptor.get_descriptors(image, kp)

        inference_filepath = self.feature_dir_path / get_inference_filename(img_filename, img_dir_rel_path)
        
        save_features(inference_filepath, kp.squeeze(0), kp_desc.squeeze(0), offset.squeeze(0))
    
    def on_validation_epoch_end(self):
        eval_task_outputs = np.concatenate(self.validation_step_outputs, axis=0)

        self.validation_step_outputs.clear()
        
        r_mAA = relative_pose_accuracy(eval_task_outputs[:, 0], self.evaluation_cfg.max_r_err_thr).mean()
        t_mAA = relative_pose_accuracy(eval_task_outputs[:, 1], self.evaluation_cfg.max_t_err_thr).mean()

        self.log('r_mAA', r_mAA)
        self.log('t_mAA', t_mAA)
        self.log('avg_mAA', (r_mAA + t_mAA) / 2)
        
        if self.descriptor is not None:
            del self.descriptor
            self.descriptor = None
            torch.cuda.empty_cache()
        
        if self.feature_dir_path is not None:
            shutil.rmtree(self.feature_dir_path)
    
    def configure_optimizers(self):
        optimizer = Adam(self.ness_detector.parameters(), lr=self.lr)
        return optimizer


"""
Support utils
"""


def kp_mask_loss(loss, kp_mask):
    if kp_mask is None:
        return loss.mean()

    else:
        num_kp = kp_mask.sum(dim=-1).float()
        batch_mask = (num_kp != 0).float()

        loss = (loss * kp_mask.float()).sum(dim=-1) / num_kp.clamp(min=1e-8)
        loss = (loss * batch_mask).sum(dim=-1) / batch_mask.sum(dim=-1).clamp(min=1e-8)

        return loss
