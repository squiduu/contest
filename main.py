import json
import multiprocessing as mp
import numpy as np
import os
import pandas as pd

from tqdm import tqdm

from common.tokenizer import tokenizer

tqdm.pandas()


def make_total_token(data):
    data["total_token"] = data["total"].progress_apply(
                                            lambda x: tok._tokenize(x)
                                            )
    return data


def make_summary_token(data):
    data["summary_token"] = data["summary"].progress_apply(
                                                lambda x: tok._tokenize(x)
                                                )
    return data


def parallel_df(df, func, n_cores):
    df_split = np.array_split(df, n_cores)
    pool = mp.Pool(n_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df


if __name__ == "__main__":
    file_dir = "./database/train/"
    file_list = os.listdir(file_dir)

    pre_data = pd.DataFrame(columns=["id", "total", "summary"])
    for file in tqdm(file_list):
        with open(os.path.join(file_dir, file), encoding="utf-8") as f:
            data = json.load(f)
            pre_data = pd.concat(
                [pre_data, pd.DataFrame(data)],
                ignore_index=True
                )

        tok = tokenizer("Kobart", "enc")

        pre_data = parallel_df(pre_data, make_total_token, n_cores=6)

        pre_data = parallel_df(pre_data, make_summary_token, n_cores=6)

        pre_data.to_json(
            os.path.join(file_dir, f"{file.replace('.json', '')}_token"),
            force_ascii=False
            )
