# HOTARU

High performance Optimizer to extract spike Timing And cell location from calcium imaging data via lineaR impUlse

### Author
TAKEKAWA Takashi <takekawa@tk2lab.org>

### Reference
- Takekawa., T, et. al.,
  HOTARU: Automatic sorting system for large scale calcium imaging data,
  bioRxiv, preparing (2022).
- Takekawa., T, et. al.,
  Automatic sorting system for large scale calcium imaging data,
  bioRxiv, https://doi.org/10.1101/215145 (2017).

### Demonstration
[Sample Video](https://drive.google.com/file/d/12jl1YTZDuNAq94ciJ-_Cj5tBcKmCqgRH/view?usp=sharing)


## Install

### Require
- python >= 3.8
- tensorflow >= 2.8

### Install Procedure (using venv)
- Create venv environment for hotaru
```sh
python3.x -m venv hotaru
```
- Activate hotaru environment
```sh
source hotaru/bin/activate
```
- Install hotaru
``sh
pip install hotaru
```


## Usage

### Config and Prepare
- Move to your workspace
```sh
cd work
```
- Edit config file `hotaru.ini`
``` hotaru.ini
[DEFAULT]
imgs_path = imgs.tif
mask_type = 0.pad
hz = 20.0
tau_rise = 0.08
tau_fall = 0.16

[main]
tag = r20

[r20]
radius_max = 20.0
```

### Check Cell Radius Stats
- Trial
```sh
hotaru trial
```
- Check peaks stats
  [see hotaru/figure/r20_trial.pdf]
- Change `radius_max` if need
``` hotaru.ini
[DEFAULT]
imgs_path = imgs.tif
mask_type = 0.pad
hz = 20.0
tau_rise = 0.08
tau_fall = 0.16

[main]
tag = r10

[r10]
radius_max = 10.0

[r20]
radius_max = 20.0
```

### Apply
- Run
```sh
hotaru auto
```
  
### Check Resutls
- see `hotaru/figure/r10_curr.pdf` and `hotaru/output` directory
