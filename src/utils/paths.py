from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent


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
