```bash
conda activate cryoet
mkdir -p data
cd data
kaggle competitions download -c czii-cryo-et-object-identification
tar -xvf czii-cryo-et-object-identification.zip
du -hs .
# 17G  .
rm czii-cryo-et-object-identification.zip
du -hs .
# 9G  .
```


Dataset knowledge:
- nb of 3D images
- nb of particles per images
- shape of 3D images

Visualizations:
- napari
- matplotlib

Embedding patches
