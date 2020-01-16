import os

def ensure_file(fpath: str, force=False) -> None:
    """Ensures the parent directory exists and the path is actually a file.

    Args:
        fpath (str): The file path
        force (bool, optional): Overwrite an existing file. Defaults to False.

    Raises:
        OSError: If the given path exists and is not a file.
        FileExistsError: If file already exists and force is set to false.
    """
    if os.path.exists(fpath):
        if not os.path.isfile(fpath):
            raise OSError("Given path '{}' is not a file".format(fpath))
        if not force:
            raise FileExistsError("File '{}' already exists".format(fpath))
        else:
            os.remove(fpath)

    os.makedirs(os.path.dirname(os.path.abspath(fpath)), exist_ok=True)
