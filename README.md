# DHRL



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


Our code sourced and modified from official implementation of [L3P](https://github.com/LunjunZhang/world-model-as-a-graph) Algorithm.
