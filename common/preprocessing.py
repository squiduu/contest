import os
import re

from soyspacing.countbase import CountSpace


class preprocessing:
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

    def spacing(self, sentence):
        return self.model.correct(sentence)

    def correct_doc(self, doc):
        """
        스페이스 교정 작업 및 문장 전처리
        """
        # Remove links
        doc = \
            re.sub(
                r"(http)([s:/]{3,4})?)?[a-z0-9]+([\.][a-z0-9]+)+[\S]+",
                " ",
                doc
                )
        # Remove E-mail
        doc = \
            re.sub(
                r"([a-zA-Z0-9_.+-]+@[a-zA-Z0-9.]+\.[a-zA-Z0-9-.]+)",
                " ",
                doc
                )
        # Remove Special char
        doc = \
            re.sub(r"[^가-힣ㄱ-ㅎㅏ-ㅣa-zA-Z0-9\s\?\!\.\-()$%+=\'\":\\]", " ", doc)
        # Remove tags
        doc = \
            re.sub(r"<(\/)?[a-z]+>", "", doc)
        # Reduce repetition
        doc = \
            re.sub(r"([^가-힣a-zA-Z0-9])\1{1,}", r"\1", doc)
        doc = \
            re.sub(r"([^0-9])\1{3,}", r"\1\1\1", doc)

        # Remove chosung
        doc = \
            re.sub(r"[ㄱ-ㅎㅏ-ㅣ]+", " ", doc)
        # Remove spaces
        doc = \
            re.sub(r"[ ]{2,}", " ", doc)
        # Arrange spaces between period
        doc = \
            re.sub(r"(\s\.\s)", ". ", doc)
        # Arrange puctuation
        doc = doc.replace(".. ", "")
        doc = doc.replace(".? ", "?")
        doc = doc.replace(".! ", "!")

        doc = doc.strip()

        # Remove line alignments
        doc = \
            re.sub(r"([\s])\1{1,}", " ", doc)
        doc = \
            re.sub(r"[\s]+", " ", doc)

        doc = doc.strip()

        # Arrange with spacing
        doc = self.model.correct(doc)

        return doc
