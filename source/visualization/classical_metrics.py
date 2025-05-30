import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from source.evaluation.logs.classical_metrics import TestClassicalMetricsLog


class ClassicalMetricsPlotter:
    VALID_METRICS = ['rep', 'mma']
    

    def __init__(self):
        mpl.rc_file('/home/konstantin/personal/Summertime/source/visualization/style.mplstyle')
    
    def plot(self,
             methods_plot_group_list: list,
             metric_name,
             informative=True,
             legend_loc_list=['best', 'best', 'best']):
        if metric_name not in self.VALID_METRICS:
            raise ValueError(f"Invalid metric: {metric_name}")

        figure_list = []

        for methods_plot_group in methods_plot_group_list:
            fig, axes = plt.subplots(1, 3, figsize=(20, 5), dpi=300)

            for axis in axes:
                axis.set_xlabel("Threshold [px]")
                axis.tick_params(which='minor', length=4, color='r')
                    
            ylabel = 'Repeatability' if metric_name == 'rep' else 'MMA'
            axes[0].set_ylabel(ylabel, fontsize=17.0)

            titles = ["Overall", "Illumination", "Viewpoint"]
            for axis, title in zip(axes, titles):
                axis.set_title(title, fontsize=17.0)

            if informative:
                fig.suptitle(methods_plot_group.name, fontsize=20, y=1.03)

            for task, alias, color in zip(methods_plot_group.tasks, methods_plot_group.aliases, methods_plot_group.colors):
                log = TestClassicalMetricsLog.from_task(task)

                if log.is_loaded():
                    all_seq, illum_seq, view_seq = log.get(metric_name)

                    px_thresh = np.linspace(1, len(all_seq), num=len(all_seq))

                    axes[0].plot(px_thresh, all_seq, label=alias, color=color)
                    axes[1].plot(px_thresh, illum_seq, label=alias, color=color)
                    axes[2].plot(px_thresh, view_seq, label=alias, color=color)

            for axis, legend_loc in zip(axes, legend_loc_list):
                axis.legend(loc=legend_loc)

            figure_list.append(fig)
            
        return figure_list
    
    def print_metrics(self, 
                      methods_plot_group_list: list,
                      metric_name,
                      threshold_idx,
                      precision=3):
        if metric_name not in self.VALID_METRICS:
            raise ValueError(f"Invalid metric: {metric_name}")
        
        for methods_plot_group in methods_plot_group_list:
            ylabel = 'Repeatability' if metric_name == 'rep' else 'MMA'
            print(f"{methods_plot_group.name}, {ylabel}")
            
            for task, alias in zip(methods_plot_group.tasks, methods_plot_group.aliases):
                log = TestClassicalMetricsLog.from_task(task)

                if log.is_loaded():
                    all_seq, illum_seq, view_seq = log.get(metric_name)
                    px_thresh = np.linspace(1, len(all_seq), num=len(all_seq))

                    print('\t', alias)
                    print('\t', "Pixel threshold:", px_thresh[threshold_idx])
                    print('\t' * 2, "Total:", f'{all_seq[threshold_idx]:.{precision}f}')
                    print('\t' * 2, "Illumination:", f'{illum_seq[threshold_idx]:.{precision}f}')
                    print('\t' * 2, "Viewpoint:", f'{view_seq[threshold_idx]:.{precision}f}', end='\n\n')
