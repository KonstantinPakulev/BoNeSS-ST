from source.utils.visualization.drawing import draw_marker, draw_covariance


class KeypointProjectionPainter:

    def __init__(self, kp_proj):
        self.kp_proj = kp_proj
    
    def get_keypoint_projections(self, batch_idx, kp_idx, patch_h_size):
        return (self.kp_proj[:, batch_idx, kp_idx] + patch_h_size).detach().cpu().numpy()
    
    def get_keypoint_projection_distribution_mean(self, batch_idx, kp_idx, patch_h_size):
        return (self.kp_proj[:, batch_idx, kp_idx].mean(dim=0) + patch_h_size).detach().cpu().numpy()
    
    def get_keypoint_projection_distribution_covariance(self, batch_idx, kp_idx):
        mean = self.kp_proj[:, batch_idx, kp_idx].mean(dim=0)

        diff = self.kp_proj[:, batch_idx, kp_idx] - mean
        cov = (diff.unsqueeze(-1) @ diff.unsqueeze(-2)).sum(dim=0) / (self.kp_proj.shape[0] - 1)

        return cov.detach().cpu().numpy()
    
    def draw_keypoint_projections(self, axis, batch_idx, kp_idx, patch_h_size, **marker_kwargs):
        kp_projs = self.get_keypoint_projections(batch_idx, kp_idx, patch_h_size)

        for kp_proj in kp_projs:
            draw_marker(axis, kp_proj, 
                        color='red', marker='x', **marker_kwargs)
    
    def draw_unimodal_approximation_mean(self, axis, batch_idx, kp_idx, patch_h_size, **marker_kwargs):
        kp_proj_dist_mean = self.get_keypoint_projection_distribution_mean(batch_idx, kp_idx, patch_h_size)

        draw_marker(axis, kp_proj_dist_mean, 
                    color='yellow', marker='+', **marker_kwargs)
    
    def draw_unimodal_approximation_covariance(self, axis, batch_idx, kp_idx, patch_h_size, **ellipse_kwargs):
        kp_proj_dist_mean = self.get_keypoint_projection_distribution_mean(batch_idx, kp_idx, patch_h_size)
        kp_proj_dist_cov = self.get_keypoint_projection_distribution_covariance(batch_idx, kp_idx)

        draw_covariance(axis, kp_proj_dist_mean, kp_proj_dist_cov, 
                        color='yellow',  zorder=2, **ellipse_kwargs)
