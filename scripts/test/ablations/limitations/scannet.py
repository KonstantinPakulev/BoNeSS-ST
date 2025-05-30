import subprocess

boness_st_e10 = [
    "+method/detector=boness_st/reference",
    "+method/descriptor=hardnet",
    "+test.evaluation.ratio_test_thr=0.92",
    "+test.evaluation.estimator.inlier_thr=2.4",
    "method.detector.checkpoint_url=file:///home/konstantin/personal/Summertime/weights/beta_ablation/boness_st/2-828/epoch=09-avg_mAA=0.7263.ckpt"
]

boness_st_e16 = [
    "+method/detector=boness_st/reference",
    "+method/descriptor=hardnet",
    "+test.evaluation.ratio_test_thr=0.92",
    "+test.evaluation.estimator.inlier_thr=2.4",
    "method.detector.checkpoint_url=file:///home/konstantin/personal/Summertime/weights/reference_model_epoch=15-avg_mAA=0.7263.ckpt"
]

overrides_list = [
    boness_st_e10,
    boness_st_e16
]


for overrides in overrides_list:    
    cmd = [
        'python', 'test.py',
        *overrides,
        '+val/dataset@test.dataset=scannet',
        '+test/experiment=default',
        '+test/evaluation=two_view_geometry/essential_matrix',
        f'+test.experiment.tags={["ablation", "limitations"]}',
        # 'test.experiment.accelerator=cpu',
        # '+mock=True',
    ]
    
    subprocess.run(cmd)
