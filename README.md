# HOTARU

High performance Optimizer to extract spike Timing And cell location from calcium imaging data via lineaR impUlse

### Author
TAKEKAWA Takashi <takekawa@tk2lab.org>

### Reference
- Takekawa., T, et. al.,
  Automatic sorting system for large scale calcium imaging data, bioRxiv (2017).
  https://doi.org/10.1101/215145


## Install

### Require
- python >= 3.8
- tensorflow >= 2.8

### Install Procedure (using venv)
- Create venv environment for hotaru
  - `python3.x -m venv hotaru`
- Activate hotaru environment
  - `source hotaru/bin/activate`
- Install hotaru
  - `pip install hotaru`


## Usage

### Prepare
- `mkdir work`
- `cd work`
- `cp somewhere/TARGET.tif imgs.tif`

### Config Option
- Set sampling rate of movie  
  `hotaru config --hz 20.0`
- Set mask file (tif or npy)  
  `hotaru config --mask-type mask.tif`
- Set calcium dynamics  
  `hotaru config --tau-rise 0.08 --tau-fall 0.16`
- Set cell size candidate  
  `hotaru config --radius-type log --radius-min 2.0 --radius-max 16.0 --radius-num 13`
  `hotaru config --radius-type linear --radius-min 2.0 --radius-max 11.0 --radius-num 10`
  `hotaru config --radius-type manual --radius-manual "2,3,4,5,6,7,8,9,10"`    

### Apply Method
- `hotaru run`
  
### Check Resutls
- see `outs` directory
- `tensorboard --logidr logs`
- open in web browser `http://localhost:6006`
