import json
import multiprocessing as mp
import numpy as np
import os
import pandas as pd

from tqdm import tqdm

from common.tokenizer import tokenizer

tqdm.pandas()


if __name__ == "__main__":
    file_dir = "./database/train/"
    file_list = os.listdir(file_dir)

    model = "KoBART"
    mode = "enc"

    tok = tokenizer(model, mode)

    for file in tqdm(file_list):
        left_file = len(file_list)
        # Get Data
        with open(os.path.join(file_dir, file), encoding="utf-8") as f:
            data = json.load(f)
        
        # Change format for tokenizing
        data = pd.DataFrame(data).to_json(orient="index", force_ascii=False)
        data = json.loads(data)

        # Preprocess and tokenizing
        for _, row in tqdm(data.items()):
            row["total_token"] = tok._tokenize(tok._preprocess(row["total"]))
            row["summary_token"] = tok._tokenize(tok._preprocess(row["summary"]))
        
        new_file_name = f"{file.replace('.json', '')}_token_{model}_{mode}.json"
        
        # Save file to Json
        with open(os.path.join(file_dir, new_file_name), "w") as f:
            json.dump(data, f)
        
        # Announcing for the number of left file
        left_file -= 1
        print(f"전처리할 파일이 {left_file}개 남았습니다.")
