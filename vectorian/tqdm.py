import os

from tqdm.auto import tqdm as tqdm_base


def tqdm(*args, **kwargs):
    level = int(os.environ.get("VECTORIAN_VERBOSE", "1"))
    if level < 1:
        kwargs["disable"] = True
    return tqdm_base(*args, **kwargs)
