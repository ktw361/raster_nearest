# Raster Triangle with Nearest Interpolation

Used for generating GT labels of uv-frag mapping.
Based on https://github.com/ethnhe/raster_triangle.

## Installation
Compile the source code by
```shell
cd src/
make
```

## Usage
python fat_rgbd_renderer.py --save-dir data/fat/uvmap --txt-file meta_data/fat_imagesets/train_merged_simple.txt

## Datasets:
- See https://github.com/NVIDIA/Dataset_Utilities for downloading .ply model.
