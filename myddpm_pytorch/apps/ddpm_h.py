


import cv2
import numpy as np
import warnings
from pathlib import Path
import os


def list_pictures(directory, ext=('jpg', 'jpeg', 'bmp', 'png', 'ppm', 'tif',
                                  'tiff')):
    """Lists all pictures in a directory, including all subdirectories.

    # Arguments
        directory: string, absolute path to the directory
        ext: tuple of strings or single string, extensions of the pictures

    # Returns
        a list of paths


    # Copy from keras_preprocessing.image.utils::list_pictures
    """
    ext = tuple('.%s' % e for e in ((ext,) if isinstance(ext, str) else ext))
    return [os.path.join(root, f)
            for root, _, files in os.walk(directory) for f in files
            if f.lower().endswith(ext)]


warnings.filterwarnings("ignore")  # 忽略keras带来的满屏警告


def create_next_version_dir(path):
    # Find all existing version directories
    version_dirs = list(Path(path).glob('version_*'))

    # If there are no existing version directories, create version_0
    if not version_dirs:
        next_version_dir = Path(path, 'version_0')
        next_version_dir.mkdir(exist_ok=True)
        return str(next_version_dir)

    # Otherwise, find the highest existing version index and increment it
    highest_index = max([int(d.name.split('_')[1]) for d in version_dirs])
    next_version_index = highest_index + 1
    next_version_dir = Path(path, f'version_{next_version_index}')
    next_version_dir.mkdir(exist_ok=True)
    return str(next_version_dir)
