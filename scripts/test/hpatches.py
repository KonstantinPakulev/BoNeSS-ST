import subprocess

sift = [
    "+method/detector=sift",
    "+method/descriptor=disk",
]

superpoint = [
    "+method/detector=superpoint",
    "+method/descriptor=disk",
]

r2d2 = [
    "+method/detector=r2d2",
    "+method/descriptor=disk",
]

keynet = [
    "+method/detector=keynet",
    "+method/descriptor=disk",
]

disk = [
    "+method/end_to_end=disk",
]

rekd = [
    "+method/detector=rekd",
    "+method/descriptor=disk",
]

shi_tomasi = [
    "+method/detector=shi_tomasi",
    "+method/descriptor=disk",
]

ness_st = [
    "+method/detector=ness_st",
    "+method/descriptor=disk",
]

boness_st = [
    "+method/detector=boness_st/reference",
    "+method/descriptor=disk",
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
        '+test/dataset=hpatches',
        '+test/experiment=default',
        '+test/evaluation=classical_metrics',
        # 'test.experiment.accelerator=cpu',
        # '+mock=True'
    ]

    subprocess.run(cmd)
