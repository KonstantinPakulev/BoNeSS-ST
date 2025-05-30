import subprocess

sift = [
    "+method/detector=sift",
    "+method/descriptor=hardnet",
    "+test.evaluation.ratio_test_thr=0.96",
    "+test.evaluation.estimator.inlier_thr=2.2"
]

superpoint = [
    "+method/detector=superpoint",
    "+method/descriptor=hardnet",
    "+test.evaluation.ratio_test_thr=0.93",
    "+test.evaluation.estimator.inlier_thr=2.8"
]

r2d2 = [
    "+method/detector=r2d2",
    "+method/descriptor=hardnet",
    "+test.evaluation.ratio_test_thr=0.92",
    "+test.evaluation.estimator.inlier_thr=2.4"
]

keynet = [
    "+method/detector=keynet",
    "+method/descriptor=hardnet",
    "+test.evaluation.ratio_test_thr=0.95",
    "+test.evaluation.estimator.inlier_thr=2.2"
]

disk = [
    "+method/detector=disk",
    "+method/descriptor=hardnet",
    "+test.evaluation.ratio_test_thr=0.93",
    "+test.evaluation.estimator.inlier_thr=2.2"
]

rekd = [
    "+method/detector=rekd",
    "+method/descriptor=hardnet",
    "+test.evaluation.ratio_test_thr=0.91",
    "+test.evaluation.estimator.inlier_thr=2.4"
]

shi_tomasi = [
    "+method/detector=shi_tomasi",
    "+method/descriptor=hardnet",
    "+test.evaluation.ratio_test_thr=0.93",
    "+test.evaluation.estimator.inlier_thr=2.4"
]

ness_st = [
    "+method/detector=ness_st",
    "+method/descriptor=hardnet",
    "+test.evaluation.ratio_test_thr=0.91",
    "+test.evaluation.estimator.inlier_thr=2.4"
]

boness_st = [
    "+method/detector=boness_st/reference",
    "+method/descriptor=hardnet",
    "+test.evaluation.ratio_test_thr=0.92",
    "+test.evaluation.estimator.inlier_thr=2.4"
]


overrides_list = [
    sift,
    superpoint,
    r2d2,
    keynet,
    disk,
    rekd,
    shi_tomasi,
    ness_st,
    boness_st,
]

for overrides in overrides_list:    
    cmd = [
        'python', 'test.py',
        *overrides,
        '+test/dataset=scannet',
        '+test/experiment=default',
        '+test/evaluation=two_view_geometry/essential_matrix',
        # 'test.experiment.accelerator=cpu',
        # '+mock=True',
    ]
    
    subprocess.run(cmd)
