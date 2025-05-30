import cv2
import numpy as np
import pydegensac
import pyopengv

from joblib import Parallel, delayed

from source.evaluation.l2_matcher import L2Matcher
from source.utils.numpy_common import angle_mat, angle_vec, normalize, to_normalized_image_coordinates, to_bearing_vector


class TwoViewGeometryEstimator:

    @staticmethod
    def from_config(estimator_cfg):
        return TwoViewGeometryEstimator(estimator_cfg.name, 
                                        estimator_cfg.inlier_thr,
                                        estimator_cfg.confidence,
                                        estimator_cfg.num_ransac_iter,
                                        estimator_cfg.min_num_matches)
    
    def __init__(self, 
                 name, 
                 inl_thr, conf, num_ransac_iter, 
                 min_num_matches):
        self.name = name

        self.inl_thr = inl_thr
        self.conf = conf
        self.num_ransac_iter = num_ransac_iter

        self.min_num_matches = min_num_matches
        
    def estimate_relative_pose(self, 
                               kp1: np.ndarray,
                               nn_kp2: np.ndarray,
                               match_mask1: np.ndarray,
                               intrinsics1: np.ndarray,
                               intrinsics2: np.ndarray):
        """
        kp1: N x 2
        nn_kp2: N x 2
        match_mask1: N
        intrinsics1: 3 x 3
        intrinsics2: 3 x 3
        """
        rel_pose, success, inl_mask = None, False, None

        if match_mask1.sum() >= self.min_num_matches:
            match_kp1, match_nn_kp2 = kp1[match_mask1], nn_kp2[match_mask1]

            if self.name == 'f_pydegensac':
                F, inl_mask = pydegensac.findFundamentalMatrix(match_kp1, match_nn_kp2,
                                                               self.inl_thr,
                                                               self.conf,
                                                               self.num_ransac_iter)
                                
                if inl_mask.sum() > 0:
                    success = True
                    rel_pose = recover_relative_pose_from_fundamental_matrix(F, inl_mask,
                                                                             match_kp1, match_nn_kp2,
                                                                             intrinsics1, intrinsics2)
                
            elif self.name == 'e_pyopengv':
                bear_vec1 = to_bearing_vector(to_normalized_image_coordinates(match_kp1, intrinsics1))
                nn_bear_vec2 = to_bearing_vector(to_normalized_image_coordinates(match_nn_kp2, intrinsics2))

                x_ang_thr1 = np.arctan2(self.inl_thr, intrinsics1[0, 0])
                y_ang_thr1 = np.arctan2(self.inl_thr, intrinsics1[1, 1])

                x_ang_thr2 = np.arctan2(self.inl_thr, intrinsics2[0, 0])
                y_ang_thr2 = np.arctan2(self.inl_thr, intrinsics2[1, 1])
                
                ang_thr = np.minimum(np.minimum(x_ang_thr1, y_ang_thr1),
                                     np.minimum(x_ang_thr2, y_ang_thr2))
                
                rel_pose = pyopengv.relative_pose_ransac(bear_vec1, nn_bear_vec2, "STEWENIUS",
                                                         1.0 - np.cos(ang_thr),
                                                         self.num_ransac_iter, self.conf)
                
                inl_mask = get_inlier_mask(bear_vec1, nn_bear_vec2, rel_pose, ang_thr)

                if inl_mask.sum() > 0:
                    success = True

                    rel_pose = pyopengv.relative_pose_optimize_nonlinear(bear_vec1[inl_mask], 
                                                                         nn_bear_vec2[inl_mask],
                                                                         rel_pose[:, 3], rel_pose[:3, :3])
                                        
                    inl_mask = get_inlier_mask(bear_vec1, 
                                               nn_bear_vec2,
                                               rel_pose, ang_thr)
                                        
                    rel_pose = np.linalg.inv(np.vstack([rel_pose, np.array([0, 0, 0, 1])]))[:3, :]

            elif self.name == 'h_pydegensac':
                H, inl_mask = pydegensac.findHomography(match_kp1, match_nn_kp2,
                                                        self.inl_thr,
                                                        self.conf,
                                                        self.num_ransac_iter)
                
                if inl_mask.sum() > 0:
                    success = True
                    rel_pose = recover_relative_pose_from_homography(H, inl_mask,
                                                                    match_kp1, match_nn_kp2,
                                                                    intrinsics1, intrinsics2)
                
            else:
                raise ValueError(f'Unknown two view geometry estimator: {self.name}')
            
        return rel_pose, success, inl_mask


"""
Relative pose metrics
"""


def relative_pose_error(rel_pose: np.ndarray, success: bool, gt_rel_pose: np.ndarray):
    """
    :param rel_pose: 3 x 4
    :param success: bool
    :param gt_rel_pose: 4 x 4
    """
    if success:
        R_err = angle_mat(rel_pose[:3, :3], gt_rel_pose[:3, :3])
        t_err = angle_vec(normalize(rel_pose[:3, 3]), normalize(gt_rel_pose[:3, 3]))

        return R_err, t_err
    
    else:
        return 180.0, 180.0


def relative_pose_error_from_extrinsics(rel_pose: np.ndarray, success: bool, extrinsics1: np.ndarray, extrinsics2: np.ndarray):
    """
    :param rel_pose: 3 x 4
    :param success: bool
    :param extrinsics1: 4 x 4
    :param extrinsics2: 4 x 4
    """
    if success:
        gt_rel_pose = extrinsics2 @ np.linalg.inv(extrinsics1)

        R_err = angle_mat(rel_pose[:3, :3], gt_rel_pose[:3, :3])
        t_err = angle_vec(normalize(rel_pose[:3, 3]), normalize(gt_rel_pose[:3, 3]))

        return R_err, t_err

    else:
        return 180.0, 180.0


def relative_pose_accuracy(err: np.ndarray, max_err_thr: int):
    """
    :param err: B x N
    """
    err_thr_list = np.linspace(1, max_err_thr, num=max_err_thr)

    accuracy = []

    for err_thr in err_thr_list:
        err_mask = (err <= err_thr).astype(np.float32)

        accuracy.append(np.mean(err_mask, axis=0))

    return np.array(accuracy)


"""
Multi-processing utils
"""


def estimate_rel_pose_error_single_pair(estimator: TwoViewGeometryEstimator, 
                                        kp1, nn_kp2, match_mask, 
                                        intrinsics1, intrinsics2, 
                                        extrinsics1=None, 
                                        extrinsics2=None,
                                        gt_rel_pose=None):
    rel_pose, success, inl_mask = estimator.estimate_relative_pose(
        kp1,
        nn_kp2,
        match_mask,
        intrinsics1,
        intrinsics2
    )

    if gt_rel_pose is None:
        R_err, t_err = relative_pose_error_from_extrinsics(rel_pose, success, 
                                                           extrinsics1, extrinsics2)
        
    else:
        R_err, t_err = relative_pose_error(rel_pose, success, gt_rel_pose)
    
    num_inl = inl_mask.sum() if success else 0

    return R_err, t_err, success, num_inl


def match_and_estimate_rel_pose_error_batch(batch, ratio_test_thr, parallelize, estimator_cfg):
    kp1, kp2 = batch['kp1'], batch['kp2']
    offset1, offset2 = batch['offset1'], batch['offset2']
    kp_desc1, kp_desc2 = batch['kp_desc1'], batch['kp_desc2']
    kp_mask1, kp_mask2 = batch['kp_mask1'], batch['kp_mask2']
    intrinsics1, intrinsics2 = batch['intrinsics1'], batch['intrinsics2']
    extrinsics1, extrinsics2 = batch.get('extrinsics1', None), batch.get('extrinsics2', None)
    rel_pose = batch.get('rel_pose', None)

    matcher = L2Matcher()

    matches = matcher.match(kp_desc1, kp_desc2, kp_mask1, kp_mask2)
    
    nn_kp2 = matches.gather_nn(kp2)
    match_mask1 = matches.get_match_mask(ratio_test_thr)

    estimator = TwoViewGeometryEstimator.from_config(estimator_cfg)

    if parallelize:
        with Parallel(n_jobs=-1) as parallel:
            results = parallel(
                delayed(estimate_rel_pose_error_single_pair)(estimator,kp1[i].cpu().numpy() + offset1[i].cpu().numpy(),
                                                            nn_kp2[i].cpu().numpy() + offset2[i].cpu().numpy(),
                                                            match_mask1[i].cpu().numpy(),
                                                            intrinsics1[i].cpu().numpy(), intrinsics2[i].cpu().numpy(),
                                                            extrinsics1[i].cpu().numpy() if extrinsics1 is not None else None,
                                                            extrinsics2[i].cpu().numpy() if extrinsics2 is not None else None,
                                                            rel_pose[i].cpu().numpy() if rel_pose is not None else None)
                                                            for i in range(kp1.shape[0]))
                                    
            return np.array(results)
        
    else:
        results = [estimate_rel_pose_error_single_pair(estimator, kp1[i].cpu().numpy() + offset1[i].cpu().numpy(),
                                                       nn_kp2[i].cpu().numpy() + offset2[i].cpu().numpy(),
                                                       match_mask1[i].cpu().numpy(),
                                                       intrinsics1[i].cpu().numpy(), intrinsics2[i].cpu().numpy(),
                                                       extrinsics1[i].cpu().numpy() if extrinsics1 is not None else None,
                                                       extrinsics2[i].cpu().numpy() if extrinsics2 is not None else None,
                                                       rel_pose[i].cpu().numpy() if rel_pose is not None else None)
                                                       for i in range(kp1.shape[0])]
        
        return np.array(results)


"""
Relative pose recovery functions
"""


def recover_relative_pose_from_fundamental_matrix(F, inl_mask,
                                                  match_kp1, match_nn_kp2,
                                                  intrinsics1, intrinsics2):
    E = intrinsics2.T @ F @ intrinsics1

    nic_kp1 = to_normalized_image_coordinates(match_kp1[inl_mask], intrinsics1)
    nic_nn_kp2 = to_normalized_image_coordinates(match_nn_kp2[inl_mask], intrinsics2)

    _, R, t, _ = cv2.recoverPose(E, nic_kp1, nic_nn_kp2)

    T = np.zeros((3, 4))
    T[:3, :3] = R
    T[:3, 3] = t.reshape(-1)

    return T


def recover_relative_pose_from_homography(H, inl_mask,
                                          match_kp1, match_nn_kp2,
                                          intrinsics1, intrinsics2):
    _, R_list, t_list, n_list = cv2.decomposeHomographyMat(np.linalg.inv(intrinsics2) @ H @ intrinsics1, np.eye(3))

    idx = cv2.filterHomographyDecompByVisibleRefpoints(R_list, n_list,
                                                       match_kp1[:, None, :], match_nn_kp2[:, None, :],
                                                       pointsMask=inl_mask.astype(np.uint8))
    
    if idx is None:
        idx = 0
    
    else:
        idx = idx[0].item()

    T = np.zeros((3, 4))
    T[:3, :3] = R_list[idx]
    T[:3, 3] = t_list[idx].reshape(-1)

    return T


def get_inlier_mask(kp_bear_vec1, kp_bear_vec2, rel_pose, ang_thr):
    R, t = rel_pose[:3, :3], rel_pose[:, 3]

    world_pt1 = pyopengv.triangulation_triangulate(kp_bear_vec1, kp_bear_vec2, t, R)
    world_pt2 = (R.T @ (world_pt1 - t[None, :]).T).T

    inl_mask1 = angle_vec(normalize(world_pt1), kp_bear_vec1, False) < ang_thr
    inl_mask2 = angle_vec(normalize(world_pt2), kp_bear_vec2, False) < ang_thr

    return inl_mask1 & inl_mask2


# num_tries = 20
# timeout = 300

# for i in range(num_tries):
#     try:
        
            
#     except Exception as e:
#         if i == num_tries - 1:
#             raise e
        
#         else:
#             print(f'Exception raised. Re-trying {i + 1}/{num_tries}...')
#             print(e)