import kss

from common.preprocessing import preprocessing
from khaiii import KhaiiiApi


class tokenizer:
    def __init__(self, model_type, mode, ):
        """
        model_type : kobart or kobertsum
        mode : enc or dec
        """
        self.model = model_type.lower()
        self.mode = mode

        # 모델별로 사용하는 토큰 구분
        if self.model == "kobart":
            self.token = {
                "bos_token": "<s>",
                "eos_token": "</s>",
                "unk_token": "<unk>",
                "pad_token": "<pad>",
                "mask_token": "<mask>",
                }
        elif self.model == "kobertsum":
            self.token = {
                "bos_token": "[CLS]",
                "eos_token": "[SEP]",
                "unk_token": "[UNK]",
                "pad_token": "[PAD]",
                "mask_token": "[MASK]",
            }
        else:
            raise ValueError("처리 불가능한 모델입니다.")


    def _preprocessing(self, doc):
        pre = preprocessing()
        doc = pre.correct_doc(doc)

        return doc

    def _tokenize(self, doc, doc_num):
        # KhaiiiApi 호출
        api = KhaiiiApi()

        # pre = preprocessing()
        # doc = pre.correct_doc(doc)

        sentences = kss.split_sentences(doc)

        # 토크나이징
        pre_sents = []
        for sentence in sentences:
            if self.mode == "dec":
                pre_sents.append(self.token["bos_token"])
            for word in api.analyze(sentence):
                for morph in word.morphs:
                    if morph.lex == ".":
                        pre_sents[-1] += morph.lex
                    else:
                        pre_sents.append(u"_" + morph.lex)
            if self.mode == "dec":
                pre_sents.append(self.token["eos_token"])

        paired = {doc_num:doc}

        return paired
