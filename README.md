# DHRL
This is a PyTorch implementation for our paper: "DHRL: A Graph-Based Approach for Long-Horizon and Sparse Hierarchical Reinforcement Learning" (NeurIPS 2022 Oral designated paper).

By [Seungjae Lee](https://jaylee0301.github.io/), [Jigang Kim](https://jigang.kim), Inkyu Jang, and H. Jin Kim

A link to our paper can be found on [arXiv](https://arxiv.org/abs/2210.05150).

## Overview

<img width="100%" src="https://user-images.githubusercontent.com/30570922/195007450-0e31b96d-798d-43ed-952e-d2ee016b8e61.JPG"/>


We present a method of Decoupling Horizons Using a Graph in Hierarchical Reinforcement Learning (DHRL) which can alleviate this problem by decoupling the horizons of high-level and low-level policies and bridging the gap between the length of both horizons using a graph. DHRL provides a freely stretchable high-level action interval, which facilitates longer temporal abstraction and faster training in complex tasks. Our method outperforms state-of-the-art HRL algorithms in typical HRL environments. Moreover, DHRL achieves long and complex locomotion and manipulation tasks.


<img width="50%" img src="https://user-images.githubusercontent.com/30570922/197715614-84a6abd0-bb7f-4bbe-9c3a-92ce78d267f0.gif"><img width="50%" img src="https://user-images.githubusercontent.com/30570922/197715628-3d6ee28b-f7dd-4d43-9f94-bb70d433ed00.gif">

<img width="25%" img src="https://user-images.githubusercontent.com/30570922/197715659-3c219c68-33b9-41e7-bb39-de192ab8502b.gif"><img width="25%" img src="https://user-images.githubusercontent.com/30570922/197715667-434be510-5778-4905-820b-9947482fa40d.gif"><img width="25%" img src="https://user-images.githubusercontent.com/30570922/197715676-c2c45312-9922-4827-9154-f53a007bdeaf.gif"><img width="25%" img src="https://user-images.githubusercontent.com/30570922/197715692-62854f86-7a54-4716-9f5d-67e934609df7.gif">


## Installation
create conda environment
```
conda create -n dhrl python=3.7
conda activate dhrl
```

install pytorch that fits your computer settings.(we used pytorch==1.7.1 and pytorch==1.11.0)
Then, install additional modules using
```
./install.sh
```

if permission denied,
```
chmod +x install.sh
chmod +x ./scripts/*.sh
```


To run MuJoCo simulation, a license is required.


## Usage
### Training and Evaluation
./scripts/{ENV}.sh {GPU} {SEED}
```
./scripts/Reacher.sh 0 0
./scripts/AntMazeSmall.sh 0 0
./scripts/AntMaze.sh 0 0
./scripts/AntMazeBottleneck.sh 0 0
./scripts/AntMazeComplex.sh 0 0
```

## Troubleshooting

protobuf error

```
pip install --upgrade protobuf==3.20.0
```

gym EntryPoint error

```
pip uninstall gym
pip install gym==0.22.0
```


## Citation
If you find this work useful in your research, please cite:
```
@inproceedings{lee2022graph,
  title={DHRL: A Graph-Based Approach for Long-Horizon and Sparse Hierarchical Reinforcement Learning},
  author={Lee, Seungjae and Kim, Jigang and Jang, Inkyu and Kim, H Jin},
  booktitle={Thirty-sixth Conference on Neural Information Processing Systems},
  pages={},
  year={2022},
  organization={}
}
```

Our code sourced and modified from official implementation of [L3P](https://github.com/LunjunZhang/world-model-as-a-graph) Algorithm.
