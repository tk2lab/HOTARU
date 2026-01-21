# HOTARU

High performance Optimizer to extract spike Timing And cell location from calcium imaging data via lineaR impUlse

### Author
TAKEKAWA Takashi <takekawa@tk2lab.org>

### References
- Takekawa., T, et. al.,
  bioRxiv, https://biorxiv.org/content/10.1101/2022.04.05.487077v2 (2023).
- Takekawa., T, et. al.,
  bioRxiv, https://biorxiv.org/content/10.1101/2022.04.05.487077v1 (2022).
- Takekawa., T, et. al.,
  bioRxiv, https://www.biorxiv.org/content/10.1101/215145 (2017).


## Install

### Require
- python >=3.11
- jax

### Recommended
- Nvidia GPU
- cuda

### Install Procedure (using uv)
- Clone git repository
```shell
git clone https://github.com/tk2lab/hotaru
```
- Setup venv
```shell
cd hotaru
uv sync
```
- Activate hotaru environment
```shell
source bin/activate
```

## Usage
see help
```shell
hotaru --help
```

## Demonstration
Download [sample.tif](https://drive.google.com/file/d/12pRyoWFRu-h15BaAAscLyoziAjiY5nP6/view?usp=drive_link)
```shell
hotaru data.imgs.file=sample.tif mode=test
# see outputs/{datetime}/test_*.pdf
hotaru data.imgs.file=sample.tif mode=run
# see outputs/{datetime}/run_*.pdf
```

[Demo Movies](https://drive.google.com/drive/folders/1yZK8vU1WOyCMuU-ogiSB7FJcZUxU8QtP?usp=sharing)
