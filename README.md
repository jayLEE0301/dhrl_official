# DHRL
This is a PyTorch implementation for our paper: "DHRL: A Graph-Based Approach for Long-Horizon and Sparse Hierarchical Reinforcement Learning" (NeurIPS 2022).

By [Seungjae Lee](https://jaylee0301.github.io/), Jigang Kim, Inkyu Jang, and H. Jin Kim

## Overview

<img width="100%" src="https://user-images.githubusercontent.com/30570922/195007450-0e31b96d-798d-43ed-952e-d2ee016b8e61.JPG"/>


We present a method of Decoupling Horizons Using a Graph in Hierarchical Reinforcement Learning (DHRL) which can alleviate this problem by decoupling the horizons of high-level and low-level policies and bridging the gap between the length of both horizons using a graph. DHRL provides a freely stretchable high-level action interval, which facilitates longer temporal abstraction and faster training in complex tasks. Our method outperforms state-of-the-art HRL algorithms in typical HRL environments. Moreover, DHRL achieves long and complex locomotion and manipulation tasks.


<img width="50%" img src="https://user-images.githubusercontent.com/30570922/195006659-d853bb54-7b3d-4c89-8069-cdeabe5d35bc.gif"><img width="50%" img src="https://user-images.githubusercontent.com/30570922/195007214-c318e489-f788-4c4b-af8e-bd5df564dca4.gif">




## Installation
```
conda create -n dhrl python=3.7
conda activate dhrl
./install.sh
```


To run MuJoCo simulation, a license is required.


## Usage
### Trainin and Evaluation
./scripts/{ENV}.sh {GPU} {SEED}
```
./scripts/Reacher.sh 0 0
./scripts/AntMazeSmall.sh 0 0
./scripts/AntMaze.sh 0 0
./scripts/AntMazeBottleneck.sh 0 0
./scripts/AntMazeComplex.sh 0 0
```

## Citation
If you find this work useful in your research, please cite:
```
@article{Lee2022dhrl,
  title={DHRL: A Graph-Based Approach for Long-Horizon and Sparse Hierarchical Reinforcement Learning},
  author={Lee, Seungjae and Kim, Jigang and Jang, Inkyu and Kim, H. Jin},
  journal={Advances in Neural Information Processing Systems},
  volume={36},
  year={2022}
}
```

Our code sourced and modified from official implementation of [L3P](https://github.com/LunjunZhang/world-model-as-a-graph) Algorithm.
