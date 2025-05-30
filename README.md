# Good Keypoints for the Two-View Geometry Estimation Problem

<div align="center" style="padding-bottom: 1em;">
<a href="https://arxiv.org/abs/2503.18767"><strong style="font-size: 1.5em;">arXiv</strong></a>
</div>

<div align="center" style="padding-bottom: 1em;">
<i>We regret to inform that the acceptance of our paper to <strong>CVPR 2025</strong> has been withdrawn due to the affiliation of some of the authors (see <a href="https://www.linkedin.com/posts/andrey-kuznetsov87_i-publish-posts-not-quite-regularly-here-activity-7321178151168991232-sF9P?utm_source=share&utm_medium=member_desktop&rcm=ACoAAAiEWF4B5-LtagPmwUDQA5XRaT9XVc6mBAk">here</a> for details).</i>
</div>

## Overview
This repository contains the reference implementation of the preprint "**Good Keypoints for the Two-View Geometry Estimation Problem**". Our work introduces **BoNeSS-ST**, a novel self-supervised keypoint detector specifically designed for the two-view geometry estimation task.

## Table of Contents
- [Setup](#setup)
- [Datasets](#datasets)
- [Training](#training)
- [Evaluation](#evaluation)
- [To-be-released](#to-be-released)
- [Citation](#citation)
## Setup


First, clone the repository:

```bash
git clone https://github.com/KonstantinPakulev/BoNeSS-ST
```

We include the repositories of the state-of-the-art keypoint detectors and descriptors as submodules in [`source/baselines`](source/baselines). If you intend on running the evaluation code or want to use them in Jupyter notebooks then you also need to initialize and update the submodules:

```bash
git submodule update --init --recursive
```

### Docker Container

For reproducibility, we provide a docker container that contains all the dependencies. Build the container with the following command:

```bash
docker compose build deploy
```

The `deploy` service contains the current copy of the code and is used for running time-consuming experiments. For visualizations and lightweight experiments, you can use the `develop` service. The services can be started with the following commands:

```bash
docker compose up deploy -d
```

```bash
docker compose up develop -d
```

We recommend to use VSCode with the **Container Tools** extension for building and running containers instead of the docker CLI.

### ClearML

We use [ClearML](https://clearml.ai/) for logging and monitoring the experiments. If you want to train or evaluate models using our code then you need to have a deployed ClearML server. Refer to the [ClearML documentation](https://clear.ml/docs/latest/docs/deploying_clearml/clearml_server_linux_mac) for the instructions about deploying the server.

Once the server is deployed, login to the server using the web interface and create a new user. Copy your credentials and use them to replace the following part in the `clearml.conf` file, which is located in the root directory of the repository:

```yaml
api {
    # Notice: 'host' is the api server (default port 8008), not the web server.
    api_server: http://10.16.112.102:8008
    web_server: http://10.16.112.102:8080/
    files_server: http://10.16.112.102:8081
    # Credentials are generated using the webapp, http://10.16.112.102:8080//settings
    # Override with os environment: CLEARML_API_ACCESS_KEY / CLEARML_API_SECRET_KEY
    credentials {
        access_key: "KR5BUYM6AIBR1YAP4PIY82K1GNPZOG"
        secret_key: "wIvySq4WwYTzJaxWX_u8vOhXFYKRdCrRDIksvT6QFYnCh7X2oeroD5o_mzftpbNQkLE"
    }
}
```

If you want to use an already running `deploy` service with the updated credentials, you need to rebuild the container and restart the service. This is due to the fact that `clearml.conf` is copied into the container during the build process.

## Datasets

For the instructions on how to download the datasets and training, validation, and testing splits, please refer to the [NeSS-ST repository](https://github.com/KonstantinPakulev/NeSS-ST).

## Training

To start the training use:

```bash
python scripts/train.py
```

Refer to the [train.ipynb](notebooks/pipeline/train.ipynb) for a detailed process of a single training iteration. 

## Evaluation

Testing scripts are located in [`scripts/test`](scripts/test).

We provide notebooks that detail the metrics that we use for the evaluation. Refer to [estimate_rel_pose_error.ipynb](notebooks/pipeline/utils/estimate_rel_pose_error.ipynb) for the details about calculating the two-view geometry estimation metrics. For classical metrics, refer to [estimate_classical_metrics.ipynb](notebooks/pipeline/utils/estimate_classical_metrics.ipynb). The way that the testing pipeline, which employs these metrics, operates is demonstrated in [two_view_geometry.ipynb](notebooks/pipeline/test/two_view_geometry.ipynb) and [classical_metrics.ipynb](notebooks/pipeline/test/classical_metrics.ipynb).

To visualize the evaluation results, refer to the [notebooks/evaluation](notebooks/evaluation).

## To-be-released
- A copy of our ClearML project with all experiments and results to enable reproducing the figures in output cells of notebooks from [notebooks/evaluation](notebooks/evaluation)
- Hyperparameter tuning pipeline

## Citation

```bibtex
@misc{pakulev2025good,
      title={Good Keypoints for the Two-View Geometry Estimation Problem}, 
      author={Konstantin Pakulev and Alexander Vakhitov and Gonzalo Ferrer},
      year={2025},
      eprint={2503.18767},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2503.18767}, 
}
```
