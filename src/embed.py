import argparse
import os
import sys
from pathlib import Path
from tqdm import tqdm
import einops

import numpy as np
import dask.array as da
import zarr
import pandas as pd

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

torch.multiprocessing.set_sharing_strategy('file_system')

tripath_path = os.environ.get('TRIPATH_PATH')
if tripath_path not in sys.path:
    sys.path.append(tripath_path)

from models.feature_extractor import resnet_3d
from models.feature_extractor import swin3d_b

CHUNK_SIZE = (30, 30, 30)
PIXEL_SIZE_FULL_RES = (1, 1, 1)
CHUNK_PHYSICAL_SIZE = np.array(PIXEL_SIZE_FULL_RES) * np.array(CHUNK_SIZE)

def open_arr(path):
    if path.endswith('.zarr'):
        store = zarr.DirectoryStore(path)
        return da.from_zarr(zarr.open(store=store, mode='r'))
    elif path.endswith('.npy'):
        img = np.load(path, mmap_mode='r')
        return da.from_array(img, chunks=CHUNK_SIZE)
    else:
        raise ValueError(f'Must be ZARR or NPY: got {path}')

def normalize_percentile(img, percentiles):
    q = np.percentile(img, percentiles)
    img = np.clip(img, q[0], q[1])
    img = img - img.min()
    img = img / (img.max() + 1) # avoid div by zero
    return img

class PatchDataset(Dataset):
    def __init__(self, paths, percentiles):
        # Load image as a Dask array
        self.path_0, self.path_1, self.path_2 = paths
        self.percentiles_0 = percentiles[0:2]
        self.percentiles_1 = percentiles[2:4]
        self.percentiles_2 = percentiles[4:6]

        self.arr_0 = open_arr(self.path_0)
        self.arr_1 = open_arr(self.path_1)
        self.arr_2 = open_arr(self.path_2)

        if not self.arr_0.shape == self.arr_1.shape == self.arr_2.shape:
            raise ValueError('All images must have the same shape')

        # Crop to multiples of CHUNK_SIZE
        self.shape = np.array(self.arr_0.shape) // CHUNK_SIZE * CHUNK_SIZE
        self.arr_0 = self.arr_0[:self.shape[0], :self.shape[1], :self.shape[2]]
        self.arr_1 = self.arr_1[:self.shape[0], :self.shape[1], :self.shape[2]]
        self.arr_2 = self.arr_2[:self.shape[0], :self.shape[1], :self.shape[2]]

        # Compute indices
        indices = np.indices(self.arr_0.blocks.shape).reshape(3, -1).T
        self.indices = [tuple(i) for i in indices]

    def __len__(self):
        return self.arr_0.blocks.size

    def __getitem__(self, idx):
        # Get the x, y, z indices of the block
        indices = self.indices[idx]

        # Load the block and normalize
        img_0 = self.arr_0.blocks[indices].compute()
        img_0 = normalize_percentile(img_0, self.percentiles_0)
        img_0 = torch.from_numpy(img_0).float() # 224 224 50

        img_1 = self.arr_1.blocks[indices].compute()
        img_1 = normalize_percentile(img_1, self.percentiles_1)
        img_1 = torch.from_numpy(img_1).float() # 224 224 50

        img_2 = self.arr_2.blocks[indices].compute()
        img_2 = normalize_percentile(img_2, self.percentiles_2)
        img_2 = torch.from_numpy(img_2).float() # 224 224 50

        # Convert to PyTorch tensor and rearrange (CZYX)
        img = einops.rearrange([img_0, img_1, img_2],
                               'c x y z -> c z y x') # 3 50 224 224
        return img, indices

def parse_args():
    parser = argparse.ArgumentParser(description='Embedding 3D patches and save to disk')
    parser.add_argument('--input', nargs=3, type=str, required=True, help='Paths to the input images')
    parser.add_argument('--output', type=str, required=True, help='Path to the output dictionary')
    parser.add_argument('--model', type=str, required=True, help='Model name. One of: resnet_3d, swin3d_b')
    parser.add_argument('--percentiles', nargs=6, type=float, required=True, help='Percentiles for normalization')
    parser.add_argument('--batch-size', type=int, required=True, help='Batch size')
    parser.add_argument('--workers', type=int, required=True, help='Number of workers')
    return parser.parse_args()

def main():
    args = parse_args()
    device = (torch.device("cuda") if torch.cuda.is_available()
              else torch.device("mps") if torch.backends.mps.is_available()
              else torch.device("cpu"))

    if args.model == 'resnet_3d':
        model = resnet_3d()
    elif args.model == 'swin3d_b':
        model = swin3d_b()
    else:
        raise ValueError(f'Invalid model: {args.model}.')
    model.load_weights()
    model = model.to(device)

    dataset = PatchDataset(args.input, args.percentiles)
    dataloader = DataLoader(dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.workers) # setting workers to 8 was important for efficiency

    with torch.no_grad():
        embeddings_list = []
        indices_list = []
        for batch, indices in tqdm(dataloader):
            batch = batch.to(device)
            indices_list.append(indices)
            embeddings = model.forward(batch).cpu()
            embeddings_list.append(embeddings)

    embeddings = torch.cat(embeddings_list) # (10140, 1024)
    indices = [torch.stack(indices).T for indices in indices_list] # nbatches 8 3
    indices = torch.cat(indices) # 10140 3

    output = dict(
        datetime = pd.Timestamp.now().isoformat(),
        embeddings=embeddings,
        indices_list=indices,
        )

    torch.save(output, args.output)

if __name__ == '__main__':
    main()
