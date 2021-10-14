import json
import multiprocessing 
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
from common.tokenizer import tokenizer


# tqdm.pandas()


summary_token = {}
total_token = {}


# def make_total_token(data):
#     data["total_token"] = data["total"].progress_apply(
#                                             lambda x: tok._tokenize(x)
#                                             )
#     return data


# def make_summary_token(data):
#     data["summary_token"] = data["summary"].progress_apply(
#                                                 lambda x: tok._tokenize(x)
#                                                 )
#     return data


def apply_async_tokenize_summary(data, tokenizer, num_cores):

    summary = data['summary']

    for id, sentence in summary.items():
        summary.update(id=tokenizer._preprocessing(sentence))
    
    p = multiprocessing.Pool(num_cores)

    for doc_num ,document in summary.items():
        p.apply_async(tokenizer._tokenize, (document, doc_num), callback=result_update)

    
    p.close()
    p.join()


def apply_async_tokenize_total(data, tokenizer, num_cores):

    total = data['total']

    for id, sententce in total.items():
        total.update(id=tokenizer._preprocessing(sententce))
    
    p = multiprocessing.Pool(num_cores)

    for doc_num ,document in total.items():
        p.apply_async(tokenizer._tokenize, (document, doc_num), callback=result_update)

    
    p.close()
    p.join()


def result_update(result):
    if result is not None:
        total_token.update(result)


# def parallel_df(df, func, n_cores):
#     df_split = np.array_split(df, n_cores)
#     pool = mp.Pool(n_cores)
#     df = pd.concat(pool.map(func, df_split))
#     pool.close()
#     pool.join()
#     return df


if __name__ == "__main__":
    file_dir = "./database/train/"
    file_list = os.listdir(file_dir)

    # pre_data = pd.DataFrame(columns=["id", "total", "summary"])
    for file in tqdm(file_list):
        with open(os.path.join(file_dir, file), encoding="utf-8") as f:
            data = json.load(f)
            # pre_data = pd.concat(
            #     [pre_data, pd.DataFrame(data)],
            #     ignore_index=True
            #     )
        # df = pd.DataFrame(columns=["id", "total", "summary"])

        tok = tokenizer("kobart", "enc")

        num_cores = multiprocessing.cpu_count()

        
        apply_async_tokenize_summary(data, tok, num_cores)
        apply_async_tokenize_total(data, tok, num_cores)

        data['total_token'] = total_token
        data['summary_token'] = summary_token


        summary_token = {}
        total_token = {}

        # df = parallel_df(df, make_total_token, n_cores=num_cores)

        # df = parallel_df(df, make_summary_token, n_cores=num_cores)

        # df.to_json(
        #     os.path.join(file_dir, f"{file.replace('.json', '')}_token"),
        #     force_ascii=False
        #     )
