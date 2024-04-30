import logging
import numpy as np
from typing import Set, List, Union, Tuple, Optional

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, LogitsProcessorList

from scripts.utils_run import FLORES101_CONVERT
from translation_models import TranslationModel
from translation_models.m2m100 import EnsembleLogitsProcessor
from translation_models.utils_llama import language_names, one_shot_sentences
logging.basicConfig(filename='translation.log', level=logging.INFO)

class LLaMaTranslationModel(TranslationModel):

    # Official templates used during instruction tuning of LLaMA
    TEMPLATE_0 = "{src_sent}\n\nTranslate to {tgt_lang}"
    TEMPLATE_1 = "{src_sent}\n\nCould you please translate this to {tgt_lang}?"
    TEMPLATE_2 = "{src_sent}\n\nTranslate this to {tgt_lang}?"
    TEMPLATE_3 = "Translate to {tgt_lang}:\n\n{src_sent}"
    TEMPLATE_4 = "Translate the following sentence to {tgt_lang}:\n{src_sent}"
    TEMPLATE_5 = "How is \"{src_sent}\" said in {tgt_lang}?"
    TEMPLATE_6 = "Translate \"{src_sent}\" to {tgt_lang}?"

    SYSTEM_PROMPT = """You are a machine translation system that translates sentences from {src_lang} to {tgt_lang}. You just respond with the translation, without any additional comments."""

    def __init__(self,
                 model_name_or_path: str,
                 message_template: str = TEMPLATE_0,
                 one_shot: bool = False,
                 padding: str = "before_system_prompt",
                 **kwargs,
                 ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map='auto', load_in_4bit=True,
                                                          torch_dtype=torch.bfloat16)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.pipeline = pipeline('text-generation', model=self.model, tokenizer=self.tokenizer)
        self.message_template = message_template
        self.one_shot = one_shot
        assert padding in ["before_system_prompt", "after_system_prompt"]
        self.padding = padding
        self.src_lang = None
        self.tgt_lang = None

    def __str__(self):
        return str(self.model_name_or_path).replace("/", "_")

    @property
    def supported_languages(self) -> Set[str]:
        return {code for code, code3 in FLORES101_CONVERT.items() if code3 in language_names}

    def requires_src_lang(self):
        return True

    def _set_src_lang(self, src_lang: str):
        assert src_lang in self.supported_languages
        self.src_lang = src_lang

    def _set_tgt_lang(self, tgt_lang: str):
        assert tgt_lang in self.supported_languages
        self.tgt_lang = tgt_lang

    def _lang_code_to_name(self, lang_code: str) -> str:
        lang_code3 = FLORES101_CONVERT.get(lang_code, lang_code)
        return language_names[lang_code3]

    @torch.no_grad()
    def _translate(self,
                   source_sentences: List[str],
                   return_score: bool = False,
                   batch_size: int = 1,
                   num_beams: int = 1,
                   **kwargs,
                   ) -> Union[List[str], List[Tuple[str, float]]]:
        if return_score:
            raise NotImplementedError
        if batch_size != 1:
            logging.warning(
                f"Batch size {batch_size} is not supported by LLaMaTranslationModel. Setting batch size to 1.")
            batch_size = 1
        if num_beams != 1:
            logging.warning(f"Beam search is not supported by LLaMaTranslationModel. Setting num_beams to 1.")
            num_beams = 1

        assert self.src_lang is not None
        assert self.tgt_lang is not None
        system_prompt = self.SYSTEM_PROMPT.format(
            src_lang=self._lang_code_to_name(self.src_lang),
            tgt_lang=self._lang_code_to_name(self.tgt_lang),
        )

        if self.one_shot:
            system_prompt += "\n\nExample instruction:\n{instruction}\n\nExample response:\nSure, here's the translation:\n{response}".format(
                instruction=self.message_template.format(
                    src_lang=self._lang_code_to_name(self.src_lang),
                    tgt_lang=self._lang_code_to_name(self.tgt_lang),
                    src_sent=one_shot_sentences[FLORES101_CONVERT.get(self.src_lang, self.src_lang)],
                ),
                response=one_shot_sentences[FLORES101_CONVERT.get(self.tgt_lang, self.tgt_lang)],
            )

        translations = []
        for source_sentence in tqdm(source_sentences):
            prompt_template = PromptTemplate(system_prompt=system_prompt)
            message = self.message_template.format(
                src_lang=self._lang_code_to_name(self.src_lang),
                tgt_lang=self._lang_code_to_name(self.tgt_lang),
                src_sent=source_sentence,
            )
            logging.info(message)
            prompt_template.add_user_message(message)
            prompt = prompt_template.build_prompt()
            prompt += "Sure, here's the translation:\n"
            inputs = self.pipeline.preprocess(prompt)
            output = self.pipeline.forward(
                inputs,
                eos_token_id=self.tokenizer.eos_token_id,
                max_length=1200,  # Max ref length across Flores-101 is 960
                remove_invalid_values=True,
                num_beams=num_beams,
                # Disable sampling
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                # manually added
                #return_dict_in_generate=True,
                #output_scores=True,
            )
            output = self.pipeline.postprocess(output)
            output = output[0]['generated_text']
            logging.info(output)
            prompt_template.add_model_reply(output, includes_history=True)
            response = prompt_template.get_model_replies(strip=True)[0]
            response_lines = response.replace("Sure, here's the translation:", "").strip().split("\n")
            if not response_lines:
                translation = ""
            else:
                translation = response_lines[0].strip()
            translations.append(translation)

        return translations

    def _translate_multi_source(self,
                                multi_source_sentences: List[str],
                                src_langs: List[str],
                                tgt_langs: List[str],
                                src_weights: Optional[List[float]] = None,
                                num_beams: int = 1,
                                **kwargs,
                                ) -> str:
        assert len(multi_source_sentences) == len(src_langs) == len(tgt_langs)
        if src_weights is not None:
            assert len(src_weights) == len(multi_source_sentences)
        if num_beams != 1:
            logging.warning(f"Beam search is not supported by LLaMaTranslationModel. Setting num_beams to 1.")
            num_beams = 1

        prompts = []
        prompt_templates = []
        for src_sent, src_lang, tgt_lang in zip(multi_source_sentences, src_langs, tgt_langs):
            system_prompt = self.SYSTEM_PROMPT.format(
                src_lang=self._lang_code_to_name(src_lang),
                tgt_lang=self._lang_code_to_name(tgt_lang),
            )
            if self.one_shot:
                system_prompt += "\n\nExample instruction:\n{instruction}\n\nExample response:\nSure, here's the translation:\n{response}".format(
                    instruction=self.message_template.format(
                        src_lang=self._lang_code_to_name(src_lang),
                        tgt_lang=self._lang_code_to_name(tgt_lang),
                        src_sent=one_shot_sentences[FLORES101_CONVERT.get(src_lang, src_lang)],
                    ),
                    response=one_shot_sentences[FLORES101_CONVERT.get(tgt_lang, tgt_lang)],
                )
            prompt_template = PromptTemplate(system_prompt=system_prompt)
            message = self.message_template.format(
                src_lang=self._lang_code_to_name(src_lang),
                tgt_lang=self._lang_code_to_name(tgt_lang),
                src_sent=src_sent,
            )
            prompt_template.add_user_message(message)
            prompt = prompt_template.build_prompt()
            prompt += "Sure, here's the translation:\n"
            prompts.append(prompt)
            prompt_templates.append(prompt_template)
        #logging.info(prompts)
        inputs = [self.pipeline.preprocess(prompt) for prompt in prompts]
        #logging.info(inputs)

        input_ids = [x['input_ids'][0].tolist() for x in inputs]
        attention_mask = [x['attention_mask'][0].tolist() for x in inputs]
        #logging.info("Input_ids before padding", input_ids, attention_mask)

        pad_token_id = self.tokenizer.get_vocab()["▁"]
        max_len = max(len(x) for x in input_ids)
        if self.padding == "before_system_prompt":
            input_ids = [[pad_token_id] * (max_len - len(x)) + x for x in input_ids]
            attention_mask = [[0] * (max_len - len(x)) + x for x in attention_mask]
        elif self.padding == "after_system_prompt":
            sys_end_id = self.tokenizer.get_vocab()[">>"]
            for i in range(len(input_ids)):
                second_inst_idx = input_ids[i].index(sys_end_id, 1)
                input_ids[i] = (input_ids[i][:second_inst_idx + 1] +
                                [pad_token_id] * (max_len - len(input_ids[i])) +
                                input_ids[i][second_inst_idx + 1:])
                attention_mask[i] = (attention_mask[i][:second_inst_idx + 1] +
                                     [0] * (max_len - len(attention_mask[i])) +
                                     attention_mask[i][second_inst_idx + 1:])

        #logging.info("Input_ids after padding", input_ids, attention_mask)
        #input_enc = self.tokenizer.batch_encode_plus(inputs, return_tensor="pt", add_special_tokens=True, truncation=True, padding=self.padding)
        input_ids = torch.tensor(input_ids).to(self.model.device)
        #print(input_ids.shape, input_ids[0].shape)
        #input_ids_de = torch.tensor(input_ids[0].unsqueeze(0)).to(self.model.device)
        #input_ids_en = torch.tensor(input_ids[1].unsqueeze(0)).to(self.model.device)
        input_ids_de = input_ids[0].unsqueeze(0).clone().detach().to(self.model.device)
        input_ids_en = input_ids[1].unsqueeze(0).clone().detach().to(self.model.device)


        attention_mask = torch.tensor(attention_mask).to(self.model.device)
        #attention_mask_de = torch.tensor(attention_mask[0].unsqueeze(0)).to(self.model.device)
        #attention_mask_en = torch.tensor(attention_mask[1].unsqueeze(0)).to(self.model.device)
        attention_mask_de = attention_mask[0].unsqueeze(0).clone().detach().to(self.model.device)
        attention_mask_en = attention_mask[1].unsqueeze(0).clone().detach().to(self.model.device)


        logits_processor = LogitsProcessorList([
            EnsembleLogitsProcessor(num_beams=num_beams, source_weights=src_weights),
        ])
        #print(logits_processor)

        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_beams=num_beams,
            eos_token_id=self.tokenizer.eos_token_id,
            max_length=1200,
            logits_processor=logits_processor,
            remove_invalid_values=True,
            # Disable sampling
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            #manually added
            return_dict_in_generate=True,
            output_scores=True,
            **kwargs,
        )
        outputs_orig = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_beams=num_beams,
            eos_token_id=self.tokenizer.eos_token_id,
            max_length=1200,
            #logits_processor=logits_processor,
            remove_invalid_values=True,
            # Disable sampling
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            # manually added
            return_dict_in_generate=True,
            output_scores=True,
            **kwargs,
        )
        """
        outputs_german = self.model.generate(
            input_ids=input_ids_de,
            attention_mask=attention_mask_de,
            num_beams=num_beams,
            eos_token_id=self.tokenizer.eos_token_id,
            max_length=1200,
            # logits_processor=logits_processor,
            remove_invalid_values=True,
            # Disable sampling
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            # manually added
            output_logits=True,
            return_dict_in_generate=True,
            output_scores=True,
            **kwargs,
        )
        """



        #logging.info(outputs)
        #logging.info(outputs_orig)

        output = outputs.sequences.reshape(1, outputs.sequences.shape[0], *outputs.sequences.shape[1:])
        first_input_id = input_ids[0]
        second_input_id = input_ids[1]

        input_length = first_input_id.shape[0]
        input_length_orig_en = second_input_id.shape[0]

        cd_tokens = outputs.sequences[0][input_length:]
        fixed_decoding_de = []
        fixed_decoding_de_trans = []
        fixed_decoding_en = []
        fixed_decoding_en_trans = []
        fixed_token = []
        for tok in cd_tokens:
            if tok == 2:
                break
            # Add the current token to the input IDs
            #print(input_ids_de.shape, attention_mask_de.shape)
            #print(input_ids_en.shape, attention_mask_en.shape)
            input_ids_de = torch.cat([input_ids_de, torch.tensor([[tok]]).to(self.model.device)], dim=1)
            input_ids_en = torch.cat([input_ids_en, torch.tensor([[tok]]).to(self.model.device)], dim=1)


            # Update the attention mask to consider the new token
            attention_mask_de = torch.cat([attention_mask_de, torch.ones_like(attention_mask_de[:, :1]).to(self.model.device)], dim=1)
            attention_mask_en = torch.cat([attention_mask_en, torch.ones_like(attention_mask_en[:, :1]).to(self.model.device)], dim=1)
            #print(input_ids_de.shape, attention_mask_de.shape)
            #print(input_ids_en.shape, attention_mask_en.shape)


            outputs_german = self.model.generate(
                input_ids=input_ids_de,
                attention_mask=attention_mask_de,
                num_beams=num_beams,
                eos_token_id=self.tokenizer.eos_token_id,
                max_length=1200,
                max_new_tokens=2,
                # logits_processor=logits_processor,
                remove_invalid_values=True,
                # Disable sampling
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                # manually added
                #output_logits=True,
                return_dict_in_generate=True,
                output_scores=True,
                **kwargs,
            )

            outputs_english = self.model.generate(
                input_ids=input_ids_en,
                attention_mask=attention_mask_en,
                num_beams=num_beams,
                eos_token_id=self.tokenizer.eos_token_id,
                max_length=1200,
                max_new_tokens=2,
                # logits_processor=logits_processor,
                remove_invalid_values=True,
                # Disable sampling
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                # manually added
                #output_logits=True,
                return_dict_in_generate=True,
                output_scores=True,
                **kwargs,
            )
            fixed_decoding_de.append(outputs_german.sequences[0][input_ids_de.shape[1]:])
            fixed_decoding_de_trans.append(self.model.compute_transition_scores(outputs_german.sequences,
                                                                                 outputs_german.scores,
                                                                                 normalize_logits=True))
            fixed_decoding_en.append(outputs_english.sequences[0][input_ids_en.shape[1]:])
            #print("transition scores before entering the list: ", self.model.compute_transition_scores(outputs_english.sequences, outputs_english.scores,
            #normalize_logits=True), outputs_english.sequences, outputs_english.scores)
            fixed_decoding_en_trans.append(
                self.model.compute_transition_scores(outputs_english.sequences, outputs_english.scores,
                                                     normalize_logits=True))
            fixed_token.append(tok)

        #logging.info(output)
        #--added start
        #originall sequence with contrastive decoding.
        transition_scores = self.model.compute_transition_scores(
            outputs.sequences, outputs.scores, normalize_logits=True)
        transition_scores_orig = self.model.compute_transition_scores(
            outputs_orig.sequences, outputs_orig.scores, normalize_logits=True)
        #transition_scores_experiment = self.model.compute_transition_scores(
            #english_translations.unsqueeze(0), english_scores, normalize_logits=True)

        #logging.info(transition_scores)


        generated_tokens = outputs.sequences[0][input_length:]
        generated_tokens_orig_de = outputs_orig.sequences[0][input_length:]
        decoded_de = self.tokenizer.decode(generated_tokens_orig_de)
        generated_tokens_orig_en = outputs_orig.sequences[1][input_length_orig_en:]
        decoded_en = self.tokenizer.decode(generated_tokens_orig_en)
        # Loop over each time step in the generated sequence

        #print(outputs.sequences[0][input_length:])
        #cd_tokens = outputs.sequences[0][input_length:]





        # Initialize an empty list to store tuple
        # s
        save_origin_probs_de = []
        save_origin_probs_en = []
        save_origin_translation = [str(decoded_de), str(decoded_en)]
        save_probs = []
        save_all_fixed_encoding_en = []
        save_all_fixed_encoding_de = []



        print("en sent with 'translate to German-English scores'...: ")
        logging.info(self.tokenizer.decode(generated_tokens))
        for tok, score in zip(generated_tokens, transition_scores[0]):
            logging.info(f"| {tok:5d} | {self.tokenizer.decode(tok):8s} | {score.cpu().numpy():.4f} | {np.exp(score.cpu().numpy()):.2%}")

            save_probs.append((int(tok.cpu()), self.tokenizer.decode(tok.cpu()), float(np.round(score.cpu().numpy(), decimals=4)), f"{np.exp(score.cpu().numpy()):.2%}"))

        print("CD base input incrementally increased (Translate to English): ")
        #print(fixed_decoding_en, fixed_decoding_en_trans)
        for idx, enc in enumerate(fixed_decoding_en):
            print("fixed up to here: ", int(fixed_token[idx].cpu()))
            #print(fixed_decoding_en_trans[idx], fixed_decoding_en_trans[idx][0])
            save_fixed_encoding_en = []
            for tok, score in zip(enc, fixed_decoding_en_trans[idx][0]):
                logging.info(
                    f"| {tok.cpu():5d} | {self.tokenizer.decode(tok):8s} | {score.cpu().numpy():.4f} | {np.exp(score.cpu().numpy()):.2%}")
                save_fixed_encoding_en.append((int(tok.cpu()), self.tokenizer.decode(tok.cpu()),
                                   float(np.round(score.cpu().numpy(), decimals=4)),
                                   f"{np.exp(score.cpu().numpy()):.2%}"))
            save_all_fixed_encoding_en.append([int(fixed_token[idx].cpu()), save_fixed_encoding_en])

        print("CD base input incrementally increased (Translate to German): ")
        for idx, enc in enumerate(fixed_decoding_de):
            print("fixed up to here: ", int(fixed_token[idx].cpu()))
            # print(fixed_decoding_en_trans[idx], fixed_decoding_en_trans[idx][0])
            save_fixed_encoding_de = []
            for tok, score in zip(enc, fixed_decoding_de_trans[idx][0]):
                logging.info(
                    f"| {tok.cpu():5d} | {self.tokenizer.decode(tok.cpu()):8s} | {score.cpu().numpy():.4f} | {np.exp(score.cpu().numpy()):.2%}")
                save_fixed_encoding_de.append((int(tok.cpu()), self.tokenizer.decode(tok.cpu()),
                                               float(np.round(score.cpu().numpy(), decimals=4)),
                                               f"{np.exp(score.cpu().numpy()):.2%}"))
            save_all_fixed_encoding_de.append([int(fixed_token[idx].cpu()), save_fixed_encoding_de])



        print("en sent with 'translate to German scores'...: ")
        logging.info(self.tokenizer.decode(generated_tokens_orig_de))
        for tok, score in zip(generated_tokens_orig_de, transition_scores_orig[0]):
            #logging.info(f"| {tok:5d} | {self.tokenizer.decode(tok):8s} | {score.cpu().numpy():.4f} | {np.exp(score.cpu().numpy()):.2%}")
            save_origin_probs_de.append((int(tok.cpu()), self.tokenizer.decode(tok.cpu()), float(np.round(score.cpu().numpy(), decimals=4)), f"{np.exp(score.cpu().numpy()):.2%}"))



        print("en sent with 'translate to English scores'...: ")
        logging.info(self.tokenizer.decode(generated_tokens_orig_en))
        for tok, score in zip(generated_tokens_orig_en, transition_scores_orig[1]):
            #logging.info(f"| {tok:5d} | {self.tokenizer.decode(tok):8s} | {score.cpu().numpy():.4f} | {np.exp(score.cpu().numpy()):.2%}")
            save_origin_probs_en.append((int(tok), self.tokenizer.decode(tok), float(np.round(score.cpu().numpy(), decimals=4)), f"{np.exp(score.cpu().numpy()):.2%}"))






        #--added end
        output = {
            "generated_sequence": output,
            "input_ids": input_ids[0],
            "prompt_text": prompts[0],
        }
        #logging.info(output)
        output = self.pipeline._ensure_tensor_on_device(output, device=torch.device("cpu"))
        output = self.pipeline.postprocess(output)
        #logging.info(output)
        output = output[0]['generated_text'] #Translate to German output
        _, output = output.rsplit("[/INST]", maxsplit=1)
        logging.info(output)

        prompt_templates[0].add_model_reply(output, includes_history=False)
        response = prompt_templates[0].get_model_replies(strip=True)[0]
        response_lines = response.replace("Sure, here's the translation:", "").strip().split("\n")
        if not response_lines:
            translation = ""
        else:
            translation = response_lines[0].strip()

        return translation, save_probs, save_origin_translation, save_origin_probs_de, save_origin_probs_en, save_all_fixed_encoding_de, save_all_fixed_encoding_en


class PromptTemplate:
    """
    Manages the conversation with a LLaMa chat model.

    Adapted from https://github.com/samrawal/llama2_chat_templater
    (c) Sam Rawal

    Adapted to be more similar to https://huggingface.co/blog/llama2#how-to-prompt-llama-2
    """

    def __init__(self, system_prompt=None, add_initial_inst=True):
        self.system_prompt = system_prompt
        self.add_initial_inst = add_initial_inst
        self.user_messages = []
        self.model_replies = []

    def add_user_message(self, message: str, return_prompt=True):
        self.user_messages.append(message)
        if return_prompt:
            return self.build_prompt()

    def add_model_reply(self, reply: str, includes_history=True, return_reply=True):
        reply_ = reply.replace(self.build_prompt(), "") if includes_history else reply
        self.model_replies.append(reply_)
        if len(self.user_messages) != len(self.model_replies):
            raise ValueError(
                "Number of user messages does not equal number of system replies."
            )
        if return_reply:
            return reply_

    def get_user_messages(self, strip=True):
        return [x.strip() for x in self.user_messages] if strip else self.user_messages

    def get_model_replies(self, strip=True):
        return [x.strip() for x in self.model_replies] if strip else self.model_replies

    def build_prompt(self):
        if len(self.user_messages) != len(self.model_replies) + 1:
            raise ValueError(
                "Error: Expected len(user_messages) = len(model_replies) + 1. Add a new user message!"
            )

        if self.system_prompt is not None:
            SYS = f"[INST] <<SYS>>\n{self.system_prompt}\n<</SYS>>"
        else:
            SYS = ""

        CONVO = ""
        SYS = "<s>" + SYS
        for i in range(len(self.user_messages) - 1):
            user_message, model_reply = self.user_messages[i], self.model_replies[i]
            conversation_ = f"{user_message} [/INST] {model_reply} </s>"
            if i != 0:
                conversation_ = "[INST] " + conversation_
            CONVO += conversation_

        if self.add_initial_inst:
            CONVO += f"[INST] {self.user_messages[-1]} [/INST]"
        else:
            if len(self.user_messages) <= 1:
                CONVO += f" {self.user_messages[-1]} [/INST]"
            else:
                raise NotImplementedError

        return SYS + CONVO
