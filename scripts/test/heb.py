import subprocess

sift = [
    "+method/detector=sift",
    "+method/descriptor=disk",
    "+test.evaluation.ratio_test_thr=1.0",
    "+test.evaluation.estimator.inlier_thr=2.6"
]

superpoint = [
    "+method/detector=superpoint",
    "+method/descriptor=disk",
    "+test.evaluation.ratio_test_thr=1.0",
    "+test.evaluation.estimator.inlier_thr=2.8"
]

r2d2 = [
    "+method/detector=r2d2",
    "+method/descriptor=disk",
    "+test.evaluation.ratio_test_thr=1.0",
    "+test.evaluation.estimator.inlier_thr=2.6"
]

keynet = [
    "+method/detector=keynet",
    "+method/descriptor=disk",
    "+test.evaluation.ratio_test_thr=1.0",
    "+test.evaluation.estimator.inlier_thr=3.0"
]

disk = [
    "+method/end_to_end=disk",
    "+test.evaluation.ratio_test_thr=0.98",
    "+test.evaluation.estimator.inlier_thr=2.6"
]

rekd = [
    "+method/detector=rekd",
    "+method/descriptor=disk",
    "+test.evaluation.ratio_test_thr=1.0",
    "+test.evaluation.estimator.inlier_thr=3.0"
]

shi_tomasi = [
    "+method/detector=shi_tomasi",
    "+method/descriptor=disk",
    "+test.evaluation.ratio_test_thr=1.0",
    "+test.evaluation.estimator.inlier_thr=2.2"
]

ness_st = [
    "+method/detector=ness_st",
    "+method/descriptor=disk",
    "+test.evaluation.ratio_test_thr=1.0",
    "+test.evaluation.estimator.inlier_thr=2.6"
]

boness_st = [
    "+method/detector=boness_st/reference",
    "+method/descriptor=disk",
    "+test.evaluation.ratio_test_thr=1.0",
    "+test.evaluation.estimator.inlier_thr=3.0"
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
        '+test/dataset=heb',
        '+test/experiment=default',
        '+test/evaluation=two_view_geometry/homography',
        # 'test.experiment.accelerator=cpu',
        # '+mock=True'
    ]

    subprocess.run(cmd)
