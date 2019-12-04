from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd


def pandas_apply_parallel(df, func, num_processes=None):
    """Apply a function on a Pandas DataFrame in parallel.

    Args:
        df (pd.core.frame.DataFrame): The dataframe
        func (function): The function to apply
        num_processes (int, optional): The number of processes to create. If None the CPU count is used. Defaults to None.

    Returns:
        pd.core.frame.DataFrame: A new dataframe with the function applied
    """
    num_processes = num_processes or cpu_count()
    with Pool(num_processes) as pool:
        splitted_df = np.array_split(df, num_processes)
        return pd.concat(pool.map(func, splitted_df))
