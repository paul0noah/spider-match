# 🕷️ SpiderMatch: 3D Shape Matching with Global Optimality and Geometric Consistency
Official repository for the CVPR 2024 best paper award candidate paper: SpiderMatch: 3D Shape Matching with Global Optimality and Geometric Consistency by Paul Roetzer and Florian Bernard (University of Bonn).
For more information, please visit our [our project page](https://paulroetzer.github.io/publications/2024-06-19-spidermatch.html).

## ⚙️ Installation
### Prerequesites
You need a working c++ compiler and cmake.
Note: builds are only tested on unix machines.

### Installation Step-by-Step

1) Create python environment
```bash 
conda create -n spidermatch python=3.8
conda activate spidermatch
conda install pytorch cudatoolkit -c pytorch # install pytorch
git clone git@github.com:paul0noah/spider-matchs.git
cd spider-match
pip install -r requirements.txt # install other necessary libraries via pip
```

2) Install sm-3dcouple (code to create the SpiderMatch integer linear program)
```bash
git clone git@github.com:paul0noah/sm-3dcouple.git
cd sm-3dcouple
python setup.py install
cd ..
```

3) Retrieve a gurobi license from the [official webpage](https://www.gurobi.com/)

## 📝 Dataset
Datasets are available from this [link](https://drive.google.com/file/d/1zbBs3NjUIBBmVebw38MC1nhu_Tpgn1gr/view?usp=share_link). Put all datasets under `./datasets/` such that the directory looks somehow like this
Two example files for `FAUST_r` shapes are included in this repository.
```bash
├── datasets
    ├── FAUST_r
    ├── SMAL_r
    ├── DT4D_r
```
We thank the original dataset providers for their contributions to the shape analysis community, and that all credits should go to the original authors.


### 🧑‍💻️‍ Usage
See `spidermatch_example.py` for example usage.

## 🚧 Troubleshooting
### Shapes not readable
There are some issues with the `.off` file format. Use e.g. meshlab to convert them to `.obj` for example

### Some libs for sm-3dcouple not found

- opengl not found:
`sudo apt-get install libglu1-mesa-dev freeglut3-dev mesa-common-dev`

- if `libxrandr` or `libxinerama` or other libs not found install them via
```bash
sudo apt-get install libxrandr-dev
sudo apt-get install libxinerama-dev
```

- if `libboost` not found install all related packages via
```bash
sudo apt-get install libboost-all-dev
```

List of potential libs not found: `libxrandr`, `libxinerama`, `libxcursor`, `libxi`, `libboost`

## 🙏 Acknowledgement
The implementation of DiffusionNet is based on [the official implementation](https://github.com/nmwsharp/diffusion-net).
The framework implementation is adapted from [Unsupervised Deep Multi Shape Matching](https://github.com/dongliangcao/Unsupervised-Deep-Multi-Shape-Matching).
This repository is adapted from [Unsupervised-Learning-of-Robust-Spectral-Shape-Matching](https://github.com/dongliangcao/Unsupervised-Learning-of-Robust-Spectral-Shape-Matching).

## 🎓Attribution
```bibtex
@inproceedings{roetzer2024spidermatch,
    author     = {Paul Roetzer and Florian Bernard},
    title     = { SpiderMatch: 3D Shape Matching with Global Optimality and Geometric Consistency },
    booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    year     = 2024
}
```

## License 🚀
This repo is licensed under MIT licence.
