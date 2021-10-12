import json
import os
import pandas as pd

from tqdm import tqdm

file_dir = "./book/"
file_list = os.listdir(file_dir)

id = 1

train = pd.DataFrame(columns=["id", "title", "context", "summary"])

for file in tqdm(file_list):
    with open(os.path.join(file_dir, file), encoding="utf-8") as f:
        train_data = json.load(f)

    train.loc[id, "id"] = train_data["passage_id"]
    train.loc[id, "title"] = train_data["metadata"]["doc_name"]
    train.loc[id, "context"] = train_data["passage"].replace("\n", " ")
    train.loc[id, "summary"] = train_data["summary"]
    train.loc[id, "total"] = \
        train_data["metadata"]["doc_name"] + " " + \
        train_data["passage"].replace("\n", " ")
    id += 1

train = train[["id", "summary", "total"]]

train.to_json("./test/book_train.json", force_ascii=False)
