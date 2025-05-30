import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from source.utils.visualization.drawing import autolabel

from source.evaluation.logs.rel_pose.test_log import TestRelPoseLog


class RelPoseAccuracyPlotter:

    def __init__(self):
        mpl.rc_file('/home/konstantin/personal/Summertime/source/visualization/style.mplstyle')
    
    def plot(self,
             methods_plot_group_list: list,
             informative=True,
             mask_fn=None,
             legend_loc_list=['best', 'best']):
        figure_list = []

        for methods_plot_group in methods_plot_group_list:
            fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=300)

            for axis in axes:
                axis.set_xlabel("Threshold [degrees]")
                axis.tick_params(which='minor', color='r')
            
            axes[0].set_ylabel("Rotation accuracy (%)", fontsize=17)
            axes[1].set_ylabel("Translation accuracy (%)", fontsize=17)

            titles = ["Rotation", "Translation"]
            for axis, title in zip(axes, titles):
                axis.set_title(title, fontsize=17.0)

            if informative:
                fig.suptitle(methods_plot_group.name, fontsize=20, y=1.03)

            for task, alias, color in zip(methods_plot_group.tasks, methods_plot_group.aliases, methods_plot_group.colors):
                log = TestRelPoseLog.from_task(task)
                
                if log.is_loaded():
                    r_acc = log.get_rotation_accuracy(mask_fn=mask_fn)
                    t_acc = log.get_translation_accuracy(mask_fn=mask_fn)

                    r_label = t_label = f"{alias}"

                    if informative:
                        r_label += f":{r_acc.mean():.3f} mAA"
                        t_label += f":{t_acc.mean():.3f} mAA"
                    
                    r_angles = np.linspace(1, len(r_acc), len(r_acc))
                    t_angles = np.linspace(1, len(t_acc), len(t_acc))

                    axes[0].plot(r_angles, r_acc * 100, label=alias, color=color)
                    axes[1].plot(t_angles, t_acc * 100, label=alias, color=color)

            for axis, legend_loc in zip(axes, legend_loc_list):
                axis.legend(loc=legend_loc)
            
            figure_list.append(fig)
            
        return figure_list
    
    def print_mAA(self, 
                  methods_plot_group_list: list,
                  precision=3,
                  mask_fn=None):
        for methods_plot_group in methods_plot_group_list:
            print(f"{methods_plot_group.name}")
            
            for task, alias in zip(methods_plot_group.tasks, methods_plot_group.aliases):
                log = TestRelPoseLog.from_task(task)

                if log.is_loaded():
                    r_mAA = log.get_rotation_mAA(mask_fn=mask_fn)
                    t_mAA = log.get_translation_mAA(mask_fn=mask_fn)

                    print('\t', f"{alias}:")
                    print('\t', f"Rotation: {r_mAA:.{precision}f} mAA", f"Translation: {t_mAA:.{precision}f} mAA", end='\n\n')


class RelPoseMeanNumInlPlotter:


    def __init__(self):
        mpl.rcdefaults()
    
    def plot(self, 
             methods_plot_group_list: list,
             mask_fn=None):
        for methods_plot_group in methods_plot_group_list:
            _, axis = plt.subplots(1, 1, figsize=(10, 5), dpi=300)

            for i, (task, alias, color) in enumerate(zip(methods_plot_group.tasks, methods_plot_group.aliases, methods_plot_group.colors)):
                log = TestRelPoseLog.from_task(task)

                if log.is_loaded():
                    mean_num_inl = log.get_mean_number_of_inliers(mask_fn=mask_fn)

                    bar = axis.bar(i * 0.25, mean_num_inl, color=color, label=alias, width=0.25)

                    autolabel(axis, bar,
                              fontsize=11.0,
                              precision=1)
            
            axis.margins(y=0.12)
            
            axis.legend()
                
    def print(self,  
              methods_plot_group_list: list,
              mask_fn=None):
        for methods_plot_group in methods_plot_group_list:
            print(methods_plot_group.name)

            for task, alias in zip(methods_plot_group.tasks, methods_plot_group.aliases):
                log = TestRelPoseLog.from_task(task)

                if log.is_loaded():
                    mean_num_inl = log.get_mean_number_of_inliers(mask_fn=mask_fn)

                    print('\t', f"{alias}:")
                    print('\t', f"Mean number of inliers: {mean_num_inl:.1f} ", end='\n\n')
