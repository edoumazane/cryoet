## CZII CryoET object identification Kaggle challenge

[Visit Challenge page](https://www.kaggle.com/competitions/czii-cryo-et-object-identification)


## Installation


```bash
# Clone repo
git clone git@github.com:edoumazane/cryoet.git
cd cryoet
cp environment_template.yaml environment.yaml # Edit the `environment.yaml` file variables with your local paths

# Create the environment `cryoet`
conda env create -f environment.yaml    # first time
conda env update -f environment.yaml    # each time you change something
conda remove -n cryoet --all            # if you want to start over

# Download the challenge data
conda activate cryoet
mkdir -p $DATA_DIR
cd $DATA_DIR
kaggle competitions download -c czii-cryo-et-object-identification
tar -xvf czii-cryo-et-object-identification.zip
du -hs .
# 17G  .
rm czii-cryo-et-object-identification.zip
du -hs .
# 9G  .

# If you use PyCharm, be aware that conda's environment variables are ignored
# However you can:
cp .env_template .env # Edit the `.env` file variables with your local paths

# For support in PyCharm's Runner
# Run / Edit configurations... / Edit configuration templates... 
# And set Paths to .env files

# For support in PyCharm's Notebooks and PyCharm's Python console: 
# run `import dotenv; dotenv.load_dotenv()` at the begin of each notebook...
```


Dataset knowledge:
- nb of 3D images
- nb of particles per images
- shape of 3D images

Visualizations:
- napari
- matplotlib

Embedding patches
