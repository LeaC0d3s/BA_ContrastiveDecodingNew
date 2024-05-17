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
    if args.cd_file:
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

    if args.base_file and args.cd05_file and args.cd09_file:

        with open(args.base_file, "r") as base:
            base_sent = json.load(base)
        with open(args.cd05_file, "r") as CD_05:
            CD_05_sent = json.load(CD_05)
        with open(args.cd09_file, "r") as CD_09:
            CD_09_sent = json.load(CD_09)
        with open("out/flores/en-de/total_ref.txt", "r") as ref:
            ref_sent = ref.readlines()

        if args.sentences:
            print("these are Baseline Sents:")
            for key in args.sentences:
                #print(f"\nThis is sentence at idx: {key}")
                print(base_sent[key][0])

            print("these are 0.5 Sents:")
            for key in args.sentences:
                print(CD_05_sent[key][0])

            print("these are 0.9 Sents:")
            for key in args.sentences:
                print(CD_09_sent[key][0])

            print("these are Ref Sents:")
            indexes = [int(i) for i in args.sentences]
            desired_lines = [ref_sent[int(i)].strip() for i in indexes if i < len(ref_sent)]
            for line in desired_lines:
                print(line)

            print("Print Sentences: Base, 0.5, 0.9, Ref: \n")
            for key, line in zip(args.sentences, desired_lines):
                print(base_sent[key][0])
                print(CD_05_sent[key][0])
                print(CD_09_sent[key][0])
                print(line)


        else:
            print("You need to add a space seperated list of sentences you want to inspect (--sentences)")
    else:
        print("You need to add a file path to Baseline, and 0.5 and 0.9 CD translations using the corresponding Command Line Argument.")








if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cd_file", type=str, default="",
                        help="The path to the CD-json file")
    parser.add_argument("--base_file", type=str, default="",
                        help="The path to the Baseline-json file")
    parser.add_argument("--cd05_file", type=str, default="",
                        help="The path to the 0.5 CD-json file")
    parser.add_argument("--cd09_file", type=str, default="",
                        help="The path to the 0.9 CD-json file")

    parser.add_argument("--cd_german_file", type=str, default="",
                        help="The path to the CD-json file containing the German parts")
    parser.add_argument("--cd_english_file", type=str, default="",
                        help="The path to the CD-json file containing the English parts")
    parser.add_argument("--language_weight", type=float, default=None,
                        help="weight of contrastive langauge used in the cd files.")
    parser.add_argument("--select_files", type=str, default="3",
                        help="Enter 3 translation files seperated by spaces. (E.g: Path/to/baseline.txt Path/to/CD_0.5.txt Path/to/CD0.9.txt")
    parser.add_argument("--sentences", type=str, nargs="+",
                        help="Optional argument, enter (space seperated) list of integers to represent the sentences you want to look at specifically.")
    args = parser.parse_args()
    main(args)