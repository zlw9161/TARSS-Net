# TARSS-Net: Temporal-Aware Radar Semantic Segmentation Network

## Paper Info
- TARSS-Net Structure
![teaser_schema](./images/teaser.png)

- TARSS-Net Basic Idea
![basic_idea](./images/basic-idea.png)

- Comparison of other sequential models
![seq_models](./images/sequential-model.png)

- Class-wise Performance
![class_wise](./images/sota-class.png)

- Input-length Performance
![input_len](./images/input-length.png)

- Results Visulization
![visulization](./images/visualization.png)


### Authors
Youcheng Zhang*, [Liwen Zhang](https://github.com/zlw9161)\*, Zijun Hu, et al., TARSS-Net: Temporal-Aware Radar Semantic Segmentation Network, accepted by NeurIPS 2024, paper ID: 9831 (* Equal Contributions).

## Basic Description

This repository contains the implementation of TARSS-Net including TARSS-Net w/ Spatio-TRAP and TARSS-Net w/ Depth-TRAP.

- TARSS-Net is developed using [MVRSS](https://github.com/valeoai/MVRSS) as basic framework, it inherits `mvrss`  model class;
- The model definition files are located in `TARSS/mvrss/models`, where `tarssnet_v1.py` defines TARSS-Net w/ Depth-TRAP (TARSS-Net\_D), `tarssnet_v2.py` defines TARSS-Net w/ Spatio-TRAP (TARSS-Net\_S);
- The detailed test results of TMVA-Net in [MVRSS-paper](https://arxiv.org/abs/2103.16214) and two TARSS-Net in submitted paper #1529 are provided at `TARSS/test_results`:
	- Results for TMVA-Net: see `TARSS/test_results/test_metrics_tmvanet.py`;
	- Results for TARSS-Net\_D: see `TARSS/test_results/test_metrics_tarrsnet_d.py`;
	- Results for TARSS-Net\_S: see `TARSS/test_results/test_metrics_tarrsnet_s.py`;

***These models are trained and tested on the [CARRADA dataset](https://arxiv.org/abs/2005.01456) under the same experimental setup in [MVRSS-paper](https://arxiv.org/abs/2103.16214).***

The CARRADA dataset is available at this link: [https://arthurouaknine.github.io/codeanddata/carrada](https://arthurouaknine.github.io/codeanddata/carrada).

## Installation

- Please first clone or download the repo from [https://anonymous.4open.science/r/TARSS-Net-3D8B](https://anonymous.4open.science/r/TARSS-Net-3D8B)

### Installation Steps
1. Install basic multi-view RSS network lib using pip:
```bash
$ cd TARSS/
$ pip install -e .
```

2. Install all the dependencies using pip or conda (taking ***pip*** as an example):
```bash
$ pip install numpy==1.20.3 Pillow>=8.1.1  
```
```bash
$ pip install scikit-image==0.18.3 scikit-learn==0.24.2 scipy==1.7.1
```
```bash
$ pip install tensorboard==2.6.0 torch==1.9.0 torchvision==0.10.0a0
```

## Running TARSS-Net

### Path Setup (Data & Logs)
1. Put the downloaded dataset "CARRADA" in your data dir, e.g., `/home/usrname/datasets`;
2. Specify the path for train/val logs, e.g., `/home/usrname/logs`;
3. Using ./utils/set_paths.py to set the data/log paths:

```bash
$ cd TARSS/mvrss/utils/
$ python set_paths.py --carrada /home/usrname/datasets --logs /home/usrnames/logs
```

### Training

Using training script `train.py` to train a model, this script will load the configuration info from a JSON config file, e.g., `config_files/tarssnet_v1.json`. The command line is as follows:

```bash
$ python train.py --cfg config_files/tarssnet_v1.json
```

**Note**: tarssnet\_v1 -> TARSS-Net w/ Depth-TRAP; tarssnet\_v2 -> TARSS-Net w/ Spatio-TRAP

### Testing

Actually, if you finished the training stage, the test results will also be calculated and saved in the log file. However, you can also just test the saved model using the script `test.py`.

Similar with the training stage, using the script `test.py` to evaluate the trained model as the following command line:

```bash
$ python test.py --cfg /home/logs/carrada/tarssnet_v1/name_of_the_model/config.json
```


## Acknowledgements
- Thank [CARRADA dataset](https://arxiv.org/abs/2005.01456) for providing the Radar dataset.
- Thank [MVRSS](https://arxiv.org/abs/2103.16214) for providing the basic model framework of multi-view RSS network. And we build our model using the basic framework of `mvrss` lib, our incremental modifications to the  `mvrss` code did not effect the TMVA-Net and MV-Net in MVRSS. 
- The paper is under review of the ACMMM2022, other special thanks will be mentioned after the final results.

## License

The TARSS-Net repo is released under the Apache 2.0 license.
