# HOTARU

High performance Optimizer to extract spike Timing And cell location from calcium imaging data via lineaR impUlse

### Author
TAKEKAWA Takashi <takekawa@tk2lab.org>

### Reference
- Takekawa., T, et. al., Automatic sorting system for large scale calcium imaging data, bioRxiv,  https://doi.org/10.1101/215145 (2017).


## Install

### Require
- python >= 3.6.1

### Install Procedure (using venv)
- Create venv environment for hotaru
  - `python3.x -m venv hotaru`
- Activate hotaru environment
  - `source hotaru/bin/activate`
- Install to hotaru
  - `pip intall hotaru-2.0.1-py3-none-any.whl`


## Simple Usage
- (in hotaru venv)
- `mkdir work`
- `cd work`
- `cp somewhere/TARGET.tif imgs.tif`
- `hotaru`
- (see out directory)

