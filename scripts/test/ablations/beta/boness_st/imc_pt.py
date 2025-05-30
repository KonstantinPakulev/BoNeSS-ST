import subprocess


boness_st = [
    "+method/detector=boness_st/default",
    "+method/descriptor=disk",
    "+test.evaluation.ratio_test_thr=0.99",
    "+test.evaluation.estimator.inlier_thr=0.7"
]

url_list = [
    "file:///home/konstantin/personal/Summertime/weights/beta_ablation/boness_st/1-189/epoch=09-avg_mAA=0.7156.ckpt",
    "file:///home/konstantin/personal/Summertime/weights/beta_ablation/boness_st/1-414/epoch=08-avg_mAA=0.7177.ckpt",
    "file:///home/konstantin/personal/Summertime/weights/beta_ablation/boness_st/1-681/epoch=07-avg_mAA=0.7223.ckpt",
    "file:///home/konstantin/personal/Summertime/weights/beta_ablation/boness_st/2-0/epoch=09-avg_mAA=0.7246.ckpt",
    "file:///home/konstantin/personal/Summertime/weights/beta_ablation/boness_st/2-378/epoch=09-avg_mAA=0.7263.ckpt",
    "file:///home/konstantin/personal/Summertime/weights/beta_ablation/boness_st/2-828/epoch=09-avg_mAA=0.7263.ckpt",
    "file:///home/konstantin/personal/Summertime/weights/beta_ablation/boness_st/3-363/epoch=08-avg_mAA=0.7241.ckpt",
    "file:///home/konstantin/personal/Summertime/weights/beta_ablation/boness_st/4-0/epoch=08-avg_mAA=0.7213.ckpt"
]

for url in url_list:
    cmd = [
        'python', 'test.py',
        '+test/dataset=imc_pt',
        '+test/experiment=default',
        '+test/evaluation=two_view_geometry/fundamental_matrix',
        *boness_st,
        f'+method.detector.checkpoint_url="{url}"',
        f'+test.experiment.tags={["ablation", "beta"]}',
        # 'test.experiment.accelerator=cpu',
        # '+mock=True'
    ]

    subprocess.run(cmd)
