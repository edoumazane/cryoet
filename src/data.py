import os

def get_experiment_paths(data_dir, experiment):
    """
    Returns a dictionary with the following keys:
        - images: dictionary with the following keys:
            - denoised: path to the denoised zarr file
            - ctfdeconvolved: path to the ctfdeconvolved zarr file
            - isonetcorrected: path to the isonetcorrected zarr file
            - wbp: path to the wbp zarr file
        - jsons: dictionary with the following keys:
            - ribosome: path to the ribosome json file
            - virus-like-particle: path to the virus-like-particle json file
            - beta-galactosidase: path to the beta-galactosidase json file
            - beta-amylase: path to the beta-amylase json file
            - apo-ferritin: path to the apo-ferritin json file
            - thyroglobulin: path to the thyroglobulin json file
    """
    images = {}
    for file_type in ["denoised", "ctfdeconvolved", "isonetcorrected", "wbp"]:
        images[file_type] = data_dir / f"train/static/ExperimentRuns/{experiment}/VoxelSpacing10.000/{file_type}.zarr"
    jsons = {}
    for file_type in ["ribosome", "virus-like-particle", "beta-galactosidase", "beta-amylase", "apo-ferritin", "thyroglobulin"]:
        jsons[file_type] = data_dir / f"train/overlay/ExperimentRuns/{experiment}/Picks/{file_type}.json"
    return dict(images=images, jsons=jsons)

def generate_experiment_list(data_dir):
    return os.listdir(data_dir / 'train/static/ExperimentRuns')

def generate_path_list(data_dir):
    experiments = generate_experiment_list(data_dir)
    return [data_dir / f'train/static/ExperimentRuns/{experiment}/VoxelSpacing10.000/denoised.zarr' for experiment in experiments]
