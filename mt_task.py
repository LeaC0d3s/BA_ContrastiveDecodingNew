import logging
import subprocess
import tempfile
import random
import copy
from pathlib import Path
from scripts.utils_run import FLORES101_CONVERT
from sacrebleu import get_source_file
from datasets import load_dataset
from tqdm import tqdm
import os
import json

class MTTask:

    def __init__(self,
                 src_lang: str,
                 tgt_lang: str,
                 testset: str,
                 ):
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.language_pair = f"{src_lang}-{tgt_lang}"
        self.testset = testset
        base_out_dir = Path(__file__).parent / "out"
        print(base_out_dir)
        assert base_out_dir.exists()
        self.out_dir = base_out_dir / self.testset
        self.out_dir.mkdir(exist_ok=True)

        self.out_dir = self.out_dir / self.language_pair
        self.out_dir.mkdir(exist_ok=True)
        self.load_converter = FLORES101_CONVERT

    def __str__(self):
        return f"{self.testset}-{self.src_lang}-{self.tgt_lang}"

    def evaluate(self, translation_method: callable, type='direct', source_contrastive=1, source_weight=None, language_contrastive=None, language_weight=None) -> Path:

        ## load FLORES dataset
        #source_sentences = load_dataset('gsarti/flores_101',self.load_converter[self.src_lang])['devtest']['sentence']

        # Define the path to your local text file
        file_path = "en_selected_sents.txt"

        # Load the dataset from the text file
        #source_sentences = load_dataset("text", data_files=file_path)["train"]["text"]

        # Open the file and read its contents
        with open(file_path, "r", encoding="utf-8") as file:
            # Read all lines from the file
            source_sentences = file.readlines()

        # Optionally, you can remove newline characters from each line
        source_sentences = [sentence.strip() for sentence in source_sentences]




        if type == 'direct':
            translations = translation_method(
            src_lang=self.src_lang,
            tgt_lang=self.tgt_lang,
            source_sentences=source_sentences,
            )
        elif type == 'contrastive':
            multi_source_sentences = [source_sentences]
            src_weights = [1]
            tgt_langs=[self.tgt_lang]
            src_langs=[self.src_lang]

            # randomly shuffled input to suppress hallucinations
            if source_contrastive:
                for i in range(source_contrastive):
                    shuffled_sentences = copy.copy(source_sentences)
                    random.shuffle(shuffled_sentences)
                    multi_source_sentences.append(shuffled_sentences)
                    src_weights.append(source_weight/source_contrastive)
                    tgt_langs.append(self.tgt_lang)
                    src_langs.append(self.src_lang)

            # input with wrong target language indicator to suppress off-target translation
            if language_contrastive:
                for offtarget in language_contrastive:
                    # ignore contrastive variants that are identical to true translation direction
                    if offtarget == self.tgt_lang:
                        continue
                    # don't create contrastive variant for src language if language is already listed (avoid duplicates)
                    if offtarget == 'src' and self.src_lang in language_contrastive:
                        continue
                    multi_source_sentences.append(source_sentences)
                    src_weights.append(language_weight)
                    if offtarget == 'src':
                        tgt_langs.append(self.src_lang)
                    else:
                        tgt_langs.append(offtarget)
                    src_langs.append(self.src_lang)

            translations = []
            translations_probs = {}
            origin_translation_probs_de = {}
            origin_translation_probs_en = {}
            for idx, pair in enumerate(tqdm(list(zip(*multi_source_sentences)))):
                translation, save_probs, save_origin_translation, save_origin_probs_de, save_origin_probs_en = translation_method(
                    src_langs=src_langs,
                    tgt_langs=tgt_langs,
                    src_weights=src_weights,
                    multi_source_sentences=pair,
                    )
                translations.append(translation)
                translations_probs[idx] = (translation, save_probs)
                origin_translation_probs_de[idx] = (save_origin_translation[0], save_origin_probs_de)
                origin_translation_probs_en[idx] = (save_origin_translation[1], save_origin_probs_en)
        else:
            raise NotImplementedError

        if type == 'direct':
            file_name = 'direct'
        elif type == 'contrastive':
            file_name = 'contrastive-{0}-{1}'.format(source_contrastive, source_weight)
            if language_contrastive:
                file_name += "-lang-{0}-{1}".format('+'.join(language_contrastive), language_weight)
        else:
            raise NotImplementedError

        with open(str(self.out_dir)+"/"+file_name+".txt", 'w') as f:
            f.write("\n".join(translations))
        with open(str(self.out_dir)+"/"+file_name+"probs_CD.json", 'w') as f:
            json.dump(translations_probs, f)

        with open(str(self.out_dir)+"/"+file_name+"probs_orig_de.json", 'w') as f:
            json.dump(origin_translation_probs_de, f)
        with open(str(self.out_dir)+"/"+file_name+"probs_orig_en.json", 'w') as f:
            json.dump(origin_translation_probs_en, f)

        if not os.path.isfile(str(self.out_dir)+"/"+"ref.text"):
            file_path = "de_selected_ref.txt"
            #target_sentences = load_dataset("text", data_files=file_path,cache_dir=None)["train"]["text"]
            #target_sentences = load_dataset('gsarti/flores_101', self.load_converter[self.tgt_lang])['devtest']['sentence']
            with open(file_path, "r", encoding="utf-8") as file:
                # Read all lines from the file
                target_sentences = file.readlines()

            # Optionally, you can remove newline characters from each line
            target_sentences = [sentence.strip() for sentence in target_sentences]

            with open(str(self.out_dir) + "/" + "ref.txt", 'w') as f:
                f.write("\n".join(target_sentences))

        return Path(f.name)
