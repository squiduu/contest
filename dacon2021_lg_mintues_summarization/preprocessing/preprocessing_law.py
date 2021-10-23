import json
import kss
import os
import pandas as pd
import sys

from tqdm import tqdm

sys.path.append("../")

from common.preprocessing import preprocessing  # noqa


if __name__ == "__main__":
    file_dir = "../database/raw/law/"
    train_dir = "../database/train/"
    file_list = os.listdir(file_dir)

    # Use preprocessing
    pre = preprocessing()

    for file in file_list:
        # Get Data
        with open(os.path.join(file_dir, file), encoding="utf-8") as f:
            data = json.load(f)

        # Change format for preprocessing
        data = pd.DataFrame(data).to_json(orient="index", force_ascii=False)
        data = json.loads(data)

        # Preprocessing law dataset
        for id, row in tqdm(data.items()):
            # Get id
            row["id"] = row["documents"]["id"]
            # Get summary data
            summary = row["documents"]["abstractive"][0]
            # Get total data
            context = ""
            for texts in row["documents"]["text"]:
                if type(texts) == list:
                    for text in texts:
                        context += text["sentence"]
            total = row["documents"]["title"] + ". " + context

            # Correct total data
            total = pre.correct_doc(total)
            summary = pre.correct_doc(summary)

            # Spacing total and summary data
            row["summary"] = kss.split_sentences(summary)
            row["total"] = kss.split_sentences(total)

        # Left columns "id", "total", "summary"
        df = pd.DataFrame(data).T
        df = df[["id", "total", "summary"]]

        # Set new file name
        new_file_name = f"{file.replace('.json', '')}_pre.json"

        # Save data to json
        df.to_json(
            os.path.join(train_dir, new_file_name),
            orient="index",
            force_ascii=False
            )
