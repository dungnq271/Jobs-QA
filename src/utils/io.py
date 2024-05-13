import os

import yaml  # type: ignore


def create_dir(dir_path):
    os.makedirs(dir_path, exist_ok=True)


def save_file(filebytes, filepath):
    with open(filepath, "wb") as f:
        f.write(filebytes)
    f.close()


def get_config(filepath):
    with open(filepath, "r") as f:
        config = yaml.safe_load(f)
    return config
