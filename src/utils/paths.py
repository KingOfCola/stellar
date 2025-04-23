from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent


def path_to_file(path: str) -> Path:
    """Returns the path to the file

    Parameters
    ----------
    path : str
        Path to the file

    Returns
    -------
    Path
        Path to the file
    """

    return ROOT_DIR / path


def data(path: str) -> Path:
    """Returns the path to the data directory

    Parameters
    ----------
    path : str
        Path to the data directory

    Returns
    -------
    Path
        Path to the data directory
    """

    return ROOT_DIR / "data" / path


def output(path: str) -> Path:
    """Returns the path to the output directory

    Parameters
    ----------
    path : str
        Path to the data directory

    Returns
    -------
    Path
        Path to the data directory
    """

    return ROOT_DIR / "output" / path
