from __future__ import print_function, division
import sys
import argparse
import json

def process_probs(key, value, cd_ger_data, cd_en_data):
    #access parts of dictionary:
    print(key)  # = Sentence index 0-1011
    # print(value)#
    print(value[0])  # decoded Translation
    # print(value[1]) # Structure: List[List[int(Tok_ID), str(Tok), float(norm_logits), str(prob_%),
                                        # List[List[int(Tok2_ID), str(Tok2), str(prob2_%)],
                                        # List[int(Tok3_ID), str(Tok3), str(prob3_%)]]
                                        # ]
    cd_ger_v = cd_ger_data[key][1]
    cd_en_v = cd_en_data[key][1]
    print(len(value[1][1:]), len(cd_ger_v), len(cd_en_v), )
    print(f"first Top_CD Generated: {value[1][:1][0][:4]}")
    prev_tok = (value[1][:1][0][0], value[1][:1][0][1])
    for tok_cd, tok_ger, tok_en in zip(value[1][1:], cd_ger_v, cd_en_v):
        print("Previous Generated Token: ", prev_tok)
        modified_cd = tok_cd[:2] + tok_cd[3:]
        print(f"Top_CD: {modified_cd[:3]}\tTop2_CD: {tok_cd[4][0]}\tTop3_CD: {tok_cd[4][1]}")
        modified_ger = tok_ger[1][0][:2] + tok_ger[1][0][3:]
        print(f"Top_ger: {modified_ger[:3]}\tTop2_ger: {tok_ger[1][0][4][0]}\tTop3_ger: {tok_ger[1][0][4][1]}")
        modified_en = tok_en[1][0][:2] + tok_en[1][0][3:]
        print(f"Top_en: {modified_en[:3]}\tTop2_en: {tok_en[1][0][4][0]}\tTop3_en: {tok_en[1][0][4][1]}")

        # print(tok_cd[4][0])
        # print(tok_cd[4][1])
        prev_tok = (tok_cd[0], tok_cd[1])

    return f"sentence {key} Completed"



def main(args):
    with open(args.cd_file, 'r') as cd_file:
        cd_data = json.load(cd_file)
    with open(args.cd_german_file, 'r') as cd_ger:
        cd_ger_data = json.load(cd_ger)
    with open(args.cd_english_file, 'r') as cd_en:
        cd_en_data = json.load(cd_en)
    if args.sentences:
        for key in args.sentences:
            value = cd_data[key]
            sentence_probs = process_probs(key,value,cd_ger_data,cd_en_data)
            print(sentence_probs)

    else:
        for key, value in cd_data.items():
            sentence_probs = process_probs(key,value,cd_ger_data,cd_en_data)
            print(sentence_probs)

            break





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cd_file", type=str, default="",
                        help="The path to the CD-json file")
    parser.add_argument("--cd_german_file", type=str, default="",
                        help="The path to the CD-json file containing the German parts")
    parser.add_argument("--cd_english_file", type=str, default="",
                        help="The path to the CD-json file containing the English parts")
    parser.add_argument("--language_weight", type=float, default=None,
                        help="weight of contrastive langauge used in the cd files.")
    parser.add_argument("--sentences", type=str, nargs="+",
                        help="Optional argument, enter (space seperated) list of integers to represent the sentences you want to look at specifically.")
    args = parser.parse_args()
    main(args)