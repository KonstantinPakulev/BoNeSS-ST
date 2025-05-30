import cv2
import numpy as np
import torch
import random
import re

from types import SimpleNamespace
from clearml import Task


METHOD_NAME2ALIAS_MAP = {
    'sift': 'SIFT',
    'superpoint': 'SuperPoint',
    'r2d2': 'R2D2',
    'keynet': 'Key.Net',
    'disk': 'DISK',
    'rekd': 'REKD',
    'shi_tomasi': 'Shi-Tomasi',
    'ness_st': 'NeSS-ST',
    'boness_st': 'BoNeSS-ST'
}

DESCRIPTOR_NAME2ALIAS_MAP = {
    'disk': 'DISK',
    'hardnet': 'HardNet',
}


def torch2cv(image: torch.FloatTensor):
    """
    :param image: C x H x W
    """
    image = image.cpu().permute(1, 2, 0).numpy()
    image = (image * 255).astype(np.uint8)
    return image


def if_torch2cv(image):
    if isinstance(image, torch.Tensor):
        return torch2cv(image)
    else:
        return image
    

def if_torch2numpy(t):
    if isinstance(t, torch.Tensor):
        return t.cpu().numpy()
    else:
        return t


def to_cv_keypoint(kp):
    """
    :param kp: N x 2
    """
    kp = list(map(lambda x: cv2.KeyPoint(x[1], x[0], 0), kp))

    return kp


def apply_follow_fn(follow_fn, kp_idx):
    kps_mask = follow_fn()
        
    kp_idx = torch.arange(kps_mask.shape[0])[kps_mask][kp_idx].item()

    print("New kp_idx: ", kp_idx)

    return kp_idx


def crop_black_area(image):
    nz_y, nz_x, _ = np.nonzero(image)

    top_left = np.array([np.min(nz_y), np.min(nz_x)])

    return image[top_left[0]:np.max(nz_y), top_left[1]:np.max(nz_x)], top_left


def get_baselines_plot_group(dataset_name: str,
                             boness_st_task_id: str,
                             boness_st_color: str = 'black',
                             tags = []):
    name_color_tuple_list = [
        ('sift', 'orange'),
        ('superpoint', 'gold'),
        ('r2d2', 'blue'),
        ('keynet', 'red'),
        ('disk', 'purple'),
        ('rekd', 'hotpink'),
        ('shi_tomasi', 'lightskyblue'),
        ('ness_st', 'cyan'),
        ('boness_st', boness_st_color)
    ]

    baselines = SimpleNamespace(
        name='baselines',
        tasks=[],
        aliases=[],
        colors=[_tuple[1] for _tuple in name_color_tuple_list]
    )

    baselines.tasks = [
        Task.get_tasks(project_name='CVPR 2025', 
                       task_name=f'^(detector|end_to_end)={_tuple[0]}',
                       tags=["__$all", f'{dataset_name}', "__$not", "ablation", *tags],
                       allow_archived=False)[0]
        for _tuple in name_color_tuple_list[:-1]
    ] + \
    [
        Task.get_task(task_id=boness_st_task_id)
    ]

    baselines.aliases = [get_alias_from_task_name(_tuple[0], task.name) for _tuple, task in zip(name_color_tuple_list, baselines.tasks)]

    return baselines


def get_ness_st_beta_ablation_plot_group(dataset_name: str):
    tags = [f'{dataset_name}', "ablation", "beta"]

    ness_st_beta_ablation = SimpleNamespace(
        name='ness_st_beta_ablation',
        tasks=Task.get_tasks(project_name='CVPR 2025', 
                            task_name='^detector=ness_st',
                            tags=["__$all", *tags],
                            allow_archived=False),
        aliases=[],
        colors=[]
    )

    ness_st_beta_ablation.aliases = [
        re.search(r'ness_st/(\d+-\d+)', task.name).group(1).replace('-', '.')
        for task in ness_st_beta_ablation.tasks
    ]

    ness_st_beta_ablation.colors = [get_random_hex_color() for _ in range(len(ness_st_beta_ablation.tasks))]

    sort_by_aliases(ness_st_beta_ablation)

    return ness_st_beta_ablation


def get_boness_st_beta_ablation_plot_group(dataset_name: str):
    boness_st_beta_ablation = SimpleNamespace(
        name='boness_st_beta_ablation',
        tasks=Task.get_tasks(project_name='CVPR 2025', 
                            task_name='^(detector|criterion)=boness_st',
                            tags=["__$all", f'{dataset_name}', "ablation", "beta"],
                            allow_archived=False),
        aliases=[],
        colors=[]
    )

    for task in boness_st_beta_ablation.tasks:
        match = re.search(r'(?:boness_st/(\d+-\d+)|beta=(\d+\.\d+))', task.name)
        matched_value = match.group(1) if match.group(1) is not None else match.group(2)
        boness_st_beta_ablation.aliases.append(matched_value.replace('-', '.'))

    boness_st_beta_ablation.colors = [get_random_hex_color() for _ in range(len(boness_st_beta_ablation.tasks))]

    sort_by_aliases(boness_st_beta_ablation)

    return boness_st_beta_ablation


def get_alias_from_task_name(method_name: str, task_name: str):
    match = re.search(r'descriptor=(\w+)', task_name)

    if match:
        return METHOD_NAME2ALIAS_MAP[method_name] + '+' + DESCRIPTOR_NAME2ALIAS_MAP[match.group(1)]
    
    else:
        return METHOD_NAME2ALIAS_MAP[method_name]


def sort_by_aliases(ablation_group):
    sorted_indices = np.argsort([float(alias) for alias in ablation_group.aliases])
    ablation_group.tasks = [ablation_group.tasks[i] for i in sorted_indices]
    ablation_group.aliases = [ablation_group.aliases[i] for i in sorted_indices]


def get_random_hex_color():
    return "#%06x" % random.randint(0, 0xFFFFFF)