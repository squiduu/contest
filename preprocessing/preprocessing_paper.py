import json
import os
import pandas as pd

from tqdm import tqdm


file_dir = "../database/train/paper/"
file_list = os.listdir(file_dir)

id = 1

for file in tqdm(file_list):
    with open(os.path.join(file_dir, file), encoding="utf-8") as f:
        train_data = json.load(f)

    train = pd.DataFrame(columns=["id", "title", "context", "summary"])

    for data in train_data["data"]:
        train.loc[id, "id"] = data["reg_no"]
        train.loc[id, "title"] = data["title"]
        train.loc[id, "context"] = data["summary_section"][0]["orginal_text"]
        train.loc[id, "summary"] = data["summary_section"][0]["summary_text"]
        train.loc[id, "total"] = \
            data["title"] + " " + data["summary_section"][0]["orginal_text"]
        id += 1

train = train[["id", "summary", "total"]]
train.to_json("./database/train/paper_train.json")
