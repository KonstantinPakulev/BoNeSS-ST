import subprocess

cmd = [
    "python", "train.py",
    "+method/detector=boness_st/default",
    "+method/descriptor=disk",
    "+train/criterion=setup/boness_st",
    "+train/dataset=megadepth",
    "+train/optimizer=default",
    "+train/experiment=default",
    "+val/dataset=imc_pt",
    "+val/evaluation=fundamental_matrix"
]

# Uncomment these lines if needed
# cmd.append("train.experiment.accelerator=cpu")
# cmd.append("+mock=True")

subprocess.run(cmd)
