import subprocess


ness_st = [
    "+method/detector=ness_st",
    "+method/descriptor=disk",
]

url_list = [
    "file:///home/konstantin/personal/Summertime/weights/beta_ablation/ness_st/1-414/model_r_mAA-t_mAA=0.7045.pt",
    "file:///home/konstantin/personal/Summertime/weights/beta_ablation/ness_st/1-681/model_r_mAA-t_mAA=0.7100.pt",
    "file:///home/konstantin/personal/Summertime/weights/beta_ablation/ness_st/2-0/model_r_mAA-t_mAA=0.7135.pt",
    "file:///home/konstantin/personal/Summertime/weights/beta_ablation/ness_st/2-378/model_r_mAA-t_mAA=0.7120.pt",
    "file:///home/konstantin/personal/Summertime/weights/beta_ablation/ness_st/2-828/model_r_mAA-t_mAA=0.7080.pt"
]

for url in url_list:
    cmd = [
        'python', 'test.py',
        *ness_st,
        '+test/dataset=hpatches',
        '+test/experiment=default',
        '+test/evaluation=classical_metrics',
        f'method.detector.checkpoint_url="{url}"',
        f'+test.experiment.tags={["ablation", "beta"]}',
        # 'test.experiment.accelerator=cpu',
        # '+mock=True'
    ]

    subprocess.run(cmd)
