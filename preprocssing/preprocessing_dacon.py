import json
import pandas as pd

from tqdm import tqdm


# 데이터 가져오기
with open("../database/train/train.json") as f:
    train_data = json.load(f)
with open("../database/test/test.json") as f:
    test_data = json.load(f)

# 학습 데이터셋 정리
train = pd.DataFrame(columns=["id", "title", "region", "context", "summary"])
uid = 1000
for data in tqdm(train_data):
    for agenda in data["context"]:
        context = ""
        for line in data["context"][agenda]:
            context += " " + data["context"][agenda][line]
        train.loc[uid, "id"] = data["id"]
        train.loc[uid, "title"] = data["title"]
        train.loc[uid, "region"] = data["region"]
        train.loc[uid, "context"] = context[:-1]
        train.loc[uid, "summary"] = data["label"][agenda]["summary"]
        train.loc[uid, "total"] = data["title"] + " " + data["region"] + " " + context[:-1]
        uid += 1

train = train[["id", "summary", "total"]]
train.to_json("../database/train/minutes_train.json")

# 학습 데이터셋 정리
test = pd.DataFrame(columns=["id", "title", "region", "context"])
uid = 2000
for data in tqdm(test_data):
    for agenda in data["context"]:
        context = ""
        for line in data["context"][agenda]:
            context += " " + data["context"][agenda][line]
        test.loc[uid, "id"] = data["id"]
        test.loc[uid, "title"] = data["title"]
        test.loc[uid, "region"] = data["region"]
        test.loc[uid, "context"] = context[:-1]
        test.loc[uid, "total"] = data["title"] + " " + data["region"] + " " + context[:-1]
        uid += 1

test = test[["id", "total"]]
test.to_json("../database/test/minutes_test.json")
