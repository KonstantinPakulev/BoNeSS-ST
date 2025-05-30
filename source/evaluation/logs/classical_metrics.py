import pandas as pd

from clearml import Task


class TestClassicalMetricsLog:

    @staticmethod
    def from_task(task):
        eval_log_path = task.artifacts['eval_task_output.csv'].get_local_copy()
        
        return TestClassicalMetricsLog( pd.read_csv(eval_log_path))
    
    def __init__(self, log_df):
        self.log_df = log_df

    def is_loaded(self):
        return self.log_df is not None
    
    def get(self, metric_name):
       illum_mask = self.log_df['image_rel_path1'].str.startswith('i')

       all_seq = self.log_df.filter(like=metric_name, axis=1)
       illum_seq = self.log_df[illum_mask].filter(like=metric_name, axis=1)
       view_seq = self.log_df[~illum_mask].filter(like=metric_name, axis=1)

       return all_seq.to_numpy().mean(axis=0), illum_seq.to_numpy().mean(axis=0), view_seq.to_numpy().mean(axis=0)
