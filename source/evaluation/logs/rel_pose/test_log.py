import pandas as pd

from clearml import Task

from source.evaluation.two_view_geometry import relative_pose_accuracy


class TestRelPoseLog:

    @staticmethod
    def from_task(task):
        eval_log_path = task.artifacts['eval_task_output.csv'].get_local_copy()
        
        return TestRelPoseLog(pd.read_csv(eval_log_path))

    def __init__(self, log_df):
        self.log_df = log_df
    
    def is_loaded(self):
        return self.log_df is not None
    
    def get_rotation_accuracy(self, max_r_err_thr=10, mask_fn=None):
        r_err = self.log_df.filter(like='R_err', axis=1)

        if mask_fn is not None:
            r_err = r_err[mask_fn(self.log_df)]
        
        return relative_pose_accuracy(r_err.to_numpy(), max_r_err_thr).mean(axis=-1)
    
    def get_translation_accuracy(self, max_t_err_thr=10, mask_fn=None):
        t_err = self.log_df.filter(like='t_err', axis=1)

        if mask_fn is not None:
            t_err = t_err[mask_fn(self.log_df)]
        
        return relative_pose_accuracy(t_err.to_numpy(), max_t_err_thr).mean(axis=-1)
    
    def get_mean_number_of_inliers(self, mask_fn=None):
        mean_num_inl = self.log_df.filter(like='num_inl', axis=1)

        if mask_fn is not None:
            mean_num_inl = mean_num_inl[mask_fn(self.log_df)]
        
        return mean_num_inl.to_numpy().mean()
    
    def get_rotation_mAA(self, max_r_err_thr=10, mask_fn=None):
        return self.get_rotation_accuracy(max_r_err_thr, mask_fn).mean()
    
    def get_translation_mAA(self, max_t_err_thr=10, mask_fn=None):
        return self.get_translation_accuracy(max_t_err_thr, mask_fn).mean()
