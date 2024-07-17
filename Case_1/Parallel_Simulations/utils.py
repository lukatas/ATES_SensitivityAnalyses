import os
import shutil
import numpy as np
from loguru import logger


def dirmaker(dird: str, erase: bool = False):
    """
    Given a folder path, check if it exists, and if not, creates it.
    :param dird: str: Directory path.
    :param erase: bool: Whether to delete existing folder or not.
    :return:
    """
    try:
        if not os.path.exists(dird):
            os.makedirs(dird)
            return 0
        else:
            if erase:
                shutil.rmtree(dird)
                os.makedirs(dird)
            return 1
    except Exception as e:
        logger.warning(e)
        return 0


def keep_essential(results_dir: str):
    """
   Deletes everything in a simulation folder except specific files.
   :param res_dir: Path to the folder containing results.
   """
    for the_file in os.listdir(results_dir):
        if (
                not the_file.endswith(".ucn")  # files we want to keep
                and not the_file.endswith(".csv")
                and not the_file.endswith(".mto")

        ):

            file_path = os.path.join(results_dir, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # difference: does not care if directory is empty or not
            except Exception as e:
                logger.warning(e)

def NewGridValues(nrow:int, ncol:int, new_value:float):
    new = np.ones((nrow, ncol)) * new_value

    return new

