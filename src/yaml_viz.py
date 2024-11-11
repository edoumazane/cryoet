import napari
import dask.array as da
import tifffile
import yaml
from pathlib import Path
import numpy as np
import argparse
import zarr
from zarr.errors import ContainsGroupError

def parse_args():
    parser = argparse.ArgumentParser(description="visualize with napari")
    parser.add_argument("yaml")
    args = parser.parse_args()
    return args

def load_yaml(path):
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return data

def load_image(path):
    extension = path.split(".")[-1]
    if extension.upper() in ["TIFF", "TIF"]:
        return da.from_zarr(tifffile.imread(path, aszarr=True))
    elif extension.upper() == "NPY":
        return da.array(np.load(path, mmap_mode="r")).swapaxes(0, 2)
    elif extension.upper() == "ZARR":
        print(path)
        try:
            return da.from_zarr(path)
        except ContainsGroupError:
            return da.from_zarr(path, component="0")
    else:
        if extension:
            raise NotImplementedError(f"Not supported extension for {path}: {extension}")
        else:
            raise Exception(f"No extension for {path}")

def views_yaml(path):
    params = load_yaml(path)
    viewers = {}
    for view in params:
        v = napari.Viewer(title=view["title"])
        for layer in view["layers"]:
            data = load_image(layer["path"])
            kwargs = {k:layer[k] for k in layer if k != "path"}
            v.add_image(data, **kwargs)
        viewers[view["title"]] = v
    return viewers

if __name__ == "__main__":
    args = parse_args()
    views_yaml(args.yaml)
    napari.run()
