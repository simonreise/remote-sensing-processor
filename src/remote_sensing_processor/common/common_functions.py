"""Common functions."""

from typing import Any

import json
import warnings
from pathlib import Path

import requests

import numpy as np


def persist(*inputs: Any) -> Any:
    """This function tries to persist an array if it is not too big to fit in memory."""
    results = []
    # Trying to persist dataset in memory (it makes processing much faster)
    for i in inputs:
        try:
            results.append(i.persist())
        except Exception:
            warnings.warn("Dataset does not fit in memory. Processing can be much slower.", stacklevel=2)
            results = inputs
            break
    # Return array instead of tuple if it consists of one element
    results = tuple(results)
    if len(results) == 1:
        results = results[0]
    return results


def create_path(path: Path) -> None:
    """Create a folder or a parent folder if input is a file."""
    if path is not None and (path.is_dir() or len(path.suffixes) == 0):
        create_folder(path, clean=False)
    else:
        create_folder(path.parent, clean=False)


def create_folder(folder: Path, clean: bool = True) -> None:
    """Create a folder or clean an existing one."""
    if not folder.exists():
        folder.mkdir(parents=True)
    else:
        if clean:
            clean_folder(folder)


def clean_folder(folder: Path) -> None:
    """Clean a folder."""
    for root, dirs, files in folder.walk(top_down=False):
        for name in files:
            (root / name).unlink()
        for name in dirs:
            (root / name).rmdir()


def delete(file: Path) -> None:
    """Delete a file or a dir."""
    if file.is_file():
        file.unlink()
    if file.is_dir():
        clean_folder(file)
        file.rmdir()


def read_json(file: Path) -> Any:
    """Read JSON from a file."""
    with file.open("r") as json_file:
        return json.load(json_file)


class NpEncoder(json.JSONEncoder):
    """Encode numpy arrays or values as JSON."""

    def default(self, obj: Any) -> Any:
        """Default behavior."""
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.dtype):
            return obj.name
        return super(NpEncoder, self).default(obj)


def write_json(obj: Any, file: Path) -> None:
    """Write an object to a JSON file."""
    with file.open("w") as json_file:
        json.dump(obj, json_file, cls=NpEncoder)


def ping(url: str) -> bool:
    """Ping a URL."""
    try:
        response = requests.head(url, timeout=10)
        return bool(response.ok)
    except Exception:
        return False
