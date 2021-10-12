import json

import pandas as pd
from khaiii import KhaiiiApi
from tqdm import tqdm


class KhaiiiTokenizer:
    # Check the progress
    tqdm.pandas()

    def __init__(self, input_file):
        self.input_file = input_file

    def tokenizer(self, input_file):
        # Assign the KhaiiiApi as api
        api = KhaiiiApi()
        # Open .json file to tokenize
        with open(input_file, encoding="utf-8") as f:
            data = json.load(f)
        # Make the data to pd.DataFrame
        df = pd.DataFrame(data)
        # Apply tokenizer
        df["total"] = df.total.progress_apply(lambda x: api.analyze(x))
        df["summary"] = df.summary.progress_apply(lambda x: api.analyze(x))
        # Save the tokenized data
        df.to_json(f"{input_file}_tokenized.json", force_ascii=False)
