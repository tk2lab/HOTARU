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
- python >= 3.6.1
- tensorflow >= 2.2.0

### Install Procedure (using venv)
- Create venv environment for hotaru
  - `python3.x -m venv hotaru`
- Activate hotaru environment
  - `source hotaru/bin/activate`
- Install hotaru
  - `pip install hotaru`


## Usage

### Apply Method
- (in hotaru venv)
- `mkdir work`
- `cd work`
- `cp somewhere/TARGET.tif imgs.tif`
- `hotaru config`
- `hotaru run`
- (see outs directory)

### Config Option
- Set sampling rate of movie  
  `hotaru config --hz 20.0`
- Set mask file (tif or npy)  
  `hotaru config --mask-type mask.tif`
- Set calcium dynamics  
  `hotaru config --tau-rise 0.08 --tau-fall 0.16`
- Set cell size candidate  
  `hotaru config --radius-type log --radius "2.0,40.0,13"`  
  `hotaru config --radius-type linear --radius "2.0,11.0,10"`  
  `hotaru config --radius-type manual --radius "2,3,4,5,6,7,8,9,10"`    
  
### Check Resutls
- (in hotaru venv and in work dir)
- `tensorboard --logidr logs`
- open in web browser `http://localhost:6006`
