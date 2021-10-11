import os

from soyspacing.countbase import CountSpace
from tqdm import tqdm

tqdm.pandas()


class spacing:
    def __init__(self, model_path, verbose, min_count, force_abs_threshold,
                 nonspace_threshold, space_threshold):
        """
        Set the parameters for spacing correction
        mode : train/inference
        """
        self.model_path = model_path
        self.verbose = verbose
        self.mc = min_count
        self.ft = force_abs_threshold
        self.nt = nonspace_threshold
        self.st = space_threshold

        self.model = CountSpace()

        if os.path.isfile(self.model_path):
            self.model.load_model(self.model_path, json_format=False)

    def train(self, corpus_path):
        self.model.train(corpus_path)
        self.model.save_model(self.model_path)
        print("모델 학습을 완료했습니다.")

    def inference(self, sentence):
        return self.model.correct(sentence)
