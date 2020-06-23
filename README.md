# HOTARU

High performance Optimizer to extract spike Timing And cell location from calcium imaging data via lineaR impUlse

### Author
TAKEKAWA Takashi <takekawa@tk2lab.org>

### Reference
- Takekawa., T, et. al.,
  Automatic sorting system for large scale calcium imaging data, bioRxiv (2017).
  https://doi.org/10.1101/215145


## Install

### Download
donwload `hotaru-3.x.y-py3-none-any.whl`
from
https://github.com/tk2lab/HOTARU/releases

### Require
- python >= 3.6.1
- tensorflow >= 2.2.0

### Install Procedure (using venv)
- Create venv environment for hotaru
  - `python3.x -m venv hotaru`
- Activate hotaru environment
  - `source hotaru/bin/activate`
- Install to hotaru
  - `pip intall hotaru-3.x.y-py3-none-any.whl`


## Usage

### Apply Method
- (in hotaru venv)
- `mkdir work`
- `cd work`
- `cp somewhere/TARGET.tif imgs.tif`
- `hotaru config`
- `hotaru config --name val` (optionaal)
- `hotaru run`
- (see outs directory)

### Check Resutls
- (in hotaru venv and in work dir)
- `tensorboard --logidr logs`
- open in web browser `http://localhost:6006`
