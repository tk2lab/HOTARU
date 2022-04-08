# HOTARU

High performance Optimizer to extract spike Timing And cell location from calcium imaging data via lineaR impUlse

### Author
TAKEKAWA Takashi <takekawa@tk2lab.org>

### References
- Takekawa., T, et. al.,
  HOTARU: Automatic sorting system for large scale calcium imaging data,
  bioRxiv, https://biorxiv.org/content/2022.04.05.487077 (2022).
- Takekawa., T, et. al.,
  Automatic sorting system for large scale calcium imaging data,
  bioRxiv, https://www.biorxiv.org/content/10.1101/215145 (2017).


## Install

### Require
- python >= 3.8
- tensorflow >= 2.8

### Install Procedure (using venv)
- Create venv environment for hotaru
```shell
python3.x -m venv hotaru
```
- Activate hotaru environment
```shell
source hotaru/bin/activate
```
- Install hotaru
```shell
pip install hotaru
```


### Demonstration
```shell
cd sample
python make.py
hotaru --tag default trial
hotaru --tag d12 trial
hotaru auto
python mpeg.py d12
```

[Sample Video](https://drive.google.com/file/d/12jl1YTZDuNAq94ciJ-_Cj5tBcKmCqgRH)


## Usage

### Config and Prepare
- Move to your workspace
```shell
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
```shell
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
```shell
hotaru auto
```
  
### Check Resutls
- see `hotaru/figure/r10_curr.pdf` and `hotaru/output` directory
