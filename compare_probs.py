from __future__ import print_function, division
import sys
import argparse
import json
from itertools import zip_longest
import csv

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

def escape_newlines(data):
    data = data.replace("\n", "\\n")
    data = data.replace("\r", "\\r")
    data = data.replace("'", "\'")
    data = data.replace('"', '\"')# "\"["
    print(data)
    return data


def process_comp_probs_reduced(key, base, cd05, cd09):


    new_line_counter_b = 0
    new_line_counter_5 = 0
    new_line_counter_9 = 0
    control_b = 0
    control_5 = 0
    control_9 = 0
    if base[key][1][0][1] != "\n":
        control_b += 1
    if cd05[key][1][0][1] != "\n":
        control_5 += 1
    if cd09[key][1][0][1] != "\n":
        control_9 += 1
    lines = []
    for b, c5, c9 in zip_longest(base[key][1], cd05[key][1], cd09[key][1], fillvalue=(0, "\n", 0, 0, [(0, "\n", 0), (0, "\n", 0)])):
        print("control: ", control_b, control_5, control_9)
        print("new line before: ", new_line_counter_b, new_line_counter_5, new_line_counter_9)
        if b[1] == "\n" or control_b == 1:
            if control_b == 1:
                control_b += 1
            new_line_counter_b = new_line_counter_b + 1
        if c5[1] == "\n" or control_5 == 1:
            if control_5 == 1:
                control_5 += 1
            new_line_counter_5 = new_line_counter_5 + 1
        if c9[1] == "\n" or control_9 == 1:
            if control_9 == 1:
                control_9 += 1
            new_line_counter_9 = new_line_counter_9 + 1
        print("new_line after: ", new_line_counter_b, new_line_counter_5, new_line_counter_9)

        if new_line_counter_b >=2 and new_line_counter_9 >= 2 and new_line_counter_5 >=2:
            break

        line = []
        if new_line_counter_b == 1:
            #print("orig b ok")
            # print: tok1_int \t tok1_txt \t tok1_% \t tok2_int \t tok2_txt \t tok2_% \t tok3_int \t tok3_txt \t tok3_%
            print(escape_newlines(f"{b[1]}\t{b[3]}\t{b[4][0][1]}\t{b[4][0][2]}\t{b[4][1][1]}\t{b[4][1][2]}\t"))
            line += [b[1], b[3], b[4][0][1], b[4][0][2], b[4][1][1], b[4][1][2]]
        if new_line_counter_b >= 2:
            #print("b ok")
            print(f"0\t0\t0\t0\t0\t0\t0")
            line += ['0', '0', '0', '0', '0', '0']

        if new_line_counter_5 == 1:
            #print("orig 5 ok")
            print(escape_newlines(f"{c5[1]}\t{c5[3]}\t{c5[4][0][1]}\t{c5[4][0][2]}\t{c5[4][1][1]}\t{c5[4][1][2]}\t"))
            line += [c5[1], c5[3], c5[4][0][1], c5[4][0][2], c5[4][1][1], c5[4][1][2]]

        if new_line_counter_5 >= 2:
            #print("5 ok")
            print(f"0\t0\t0\t0\t0\t0\t")
            line += ['0', '0', '0', '0', '0', '0']

        if new_line_counter_9 == 1:
            #print("orig 9 ok")
            print(escape_newlines(f"{c9[1]}\t{c9[3]}\t{c9[4][0][1]}\t{c9[4][0][2]}\t{c9[4][1][1]}\t{c9[4][1][2]}"))
            line += [c9[1], c9[3], c9[4][0][1], c9[4][0][2], c9[4][1][1], c9[4][1][2]]
        if new_line_counter_9 >= 2:
            #print("9 ok")
            print(f"0\t0\t0\t0\t0\t0")
            line += ['0', '0', '0', '0', '0', '0']

        lines.append(line)

    return lines

def process_comp_probs(key, base, cd05, cd09):

    new_line_counter_b = 0
    new_line_counter_5 = 0
    new_line_counter_9 = 0
    control_b = 0
    control_5 = 0
    control_9 = 0
    if base[key][1][0][1] != "\n":
        control_b += 1
    if cd05[key][1][0][1] != "\n":
        control_5 += 1
    if cd09[key][1][0][1] != "\n":
        control_9 += 1
    lines = []
    for b, c5, c9 in zip_longest(base[key][1],cd05[key][1], cd09[key][1], fillvalue=(0, "\n", 0, 0, [(0, "\n", 0), (0, "\n", 0)])):
        print("control: ", control_b, control_5, control_9)
        print("new line before: ", new_line_counter_b, new_line_counter_5, new_line_counter_9)

        if b[1] == "\n" or control_b == 1:
            if control_b == 1:
                control_b += 1
            new_line_counter_b += 1
        if c5[1] == "\n" or control_5 == 1:
            if control_5 == 1:
                control_5 += 1
            new_line_counter_5 += 1
        if c9[1] == "\n" or control_9 == 1:
            if control_9 == 1:
                control_9 += 1
            new_line_counter_9 += 1

        print("new_line after: ",new_line_counter_b, new_line_counter_5, new_line_counter_9)
        line = []

        if new_line_counter_b >=2 and new_line_counter_9 >= 2 and new_line_counter_5 >=2:
            break

        if new_line_counter_b == 1:
            #print("orig b ok")
            #print: tok1_int \t tok1_txt \t tok1_% \t tok2_int \t tok2_txt \t tok2_% \t tok3_int \t tok3_txt \t tok3_%
            print(escape_newlines(f"{b[0]}\t{b[1]}\t{b[3]}\t{b[4][0][0]}\t{b[4][0][1]}\t{b[4][0][2]}\t{b[4][1][0]}\t{b[4][1][1]}\t{b[4][1][2]}\t"))
            line += [b[0], b[1], b[3], b[4][0][0], b[4][0][1], b[4][0][2], b[4][1][0], b[4][1][1], b[4][1][2]]

        if new_line_counter_b >= 2:
            #print("b ok")
            print(f"\t\t\t\t\t\t\t\t\t")
            line += ['0', '0', '0', '0', '0', '0', '0', '0', '0']

        if new_line_counter_5 == 1:
            #print("orig 5 ok")
            print(escape_newlines(f"{c5[0]}\t{c5[1]}\t{c5[3]}\t{c5[4][0][0]}\t{c5[4][0][1]}\t{c5[4][0][2]}\t{c5[4][1][0]}\t{c5[4][1][1]}\t{c5[4][1][2]}\t"))
            line += [c5[0], c5[1], c5[3], c5[4][0][0], c5[4][0][1], c5[4][0][2], c5[4][1][0], c5[4][1][1], c5[4][1][2]]


        if new_line_counter_5 >= 2:
            #print("5 ok")
            print(f"\t\t\t\t\t\t\t\t\t")
            line += ['0', '0', '0', '0', '0', '0', '0', '0', '0']

        if new_line_counter_9 == 1:
            #print("orig 9 ok")
            print(escape_newlines(f"{c9[0]}\t{c9[1]}\t{c9[3]}\t{c9[4][0][0]}\t{c9[4][0][1]}\t{c9[4][0][2]}\t{c9[4][1][0]}\t{c9[4][1][1]}\t{c9[4][1][2]}"))
            line += [c9[0], c9[1], c9[3], c9[4][0][0], c9[4][0][1], c9[4][0][2], c9[4][1][0], c9[4][1][1], c9[4][1][2]]

        if new_line_counter_9 >= 2:
            #print("9 ok")
            print(f"\t\t\t\t\t\t\t\t\t")
            line += ['0', '0', '0', '0', '0', '0', '0', '0', '0']


        lines.append(line)
    return lines






def main(args):
    if args.cd_file:
        with open(args.cd_file, 'r', encoding="utf-8") as cd_file:
            cd_data = json.load(cd_file)
        with open(args.cd_german_file, 'r', encoding="utf-8") as cd_ger:
            cd_ger_data = json.load(cd_ger)
        with open(args.cd_english_file, 'r', encoding="utf-8") as cd_en:
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

        with open(args.base_file, "r", encoding="utf-8") as base:
            base_sent = json.load(base)
        with open(args.cd05_file, "r", encoding="utf-8") as CD_05:
            CD_05_sent = json.load(CD_05)
        with open(args.cd09_file, "r", encoding="utf-8") as CD_09:
            CD_09_sent = json.load(CD_09)
        with open("out/flores/en-de/total_ref.txt", "r", encoding="utf-8") as ref:
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


            with open("prob_comp_table.csv", "w", encoding="utf-8") as file:
                writer = csv.writer(file, delimiter='\t')
                writer.writerow(["Filter including top 3 most probable tokens at each generation step"])
                writer.writerow(["Baseline",'', '', '', '', '', '', '', '', '',"CD:05",'', '', '', '', '', '', '', '', '',"CD:09"])
                for key in args.sentences:
                    writer.writerow([f'Idx: {key}','', '', '', '', '', '', '', '', f'Idx: {key}', '', '', '', '', '', '', '', '', f'Idx: {key}'])

                    writer.writerow(['tok1_int', 'tok1_txt', 'tok1_%',
                                'tok2_int', 'tok2_txt', 'tok2_%',
                                'tok3_int', 'tok3_txt', 'tok3_%',
                                'tok1_int', 'tok1_txt', 'tok1_%',
                                'tok2_int', 'tok2_txt', 'tok2_%',
                                'tok3_int', 'tok3_txt', 'tok3_%',
                                'tok1_int', 'tok1_txt', 'tok1_%',
                                'tok2_int', 'tok2_txt', 'tok2_%',
                                'tok3_int', 'tok3_txt', 'tok3_%'])
                    comp_probs = process_comp_probs(key, base_sent, CD_05_sent, CD_09_sent)
                    writer.writerows(comp_probs)


            with open("prob_comp_table_redu.csv", "w", encoding="utf-8") as file:
                writer = csv.writer(file, delimiter='\t')
                writer.writerow(['Filter including top 3 most probable tokens at each generation step'])
                writer.writerow(['Baseline', '', '', '', '', '', 'CD:05', '', '', '', '', '', 'CD:09', '', '', '', '', ''])
                for key in args.sentences:
                    writer.writerow([f'Idx: {key}','','','','','',f'Idx: {key}','','','','','',f'Idx: {key}','','','','',''])
                    writer.writerow(['tok1_txt', 'tok1_%', 'tok2_txt', 'tok2_%', 'tok3_txt', 'tok3_%', 'tok1_txt', 'tok1_%t', 'tok2_txt', 'tok2_%', 'tok3_txt', 'tok3_%', 'tok1_txt', 'tok1_%', 'tok2_txt', 'tok2_%', 'tok3_txt', 'tok3_%'])

                    comp_probs = process_comp_probs_reduced(key, base_sent, CD_05_sent, CD_09_sent)
                    writer.writerows(comp_probs)



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
                        help="The path to the CD-json file containing the German parts of the CD split")
    parser.add_argument("--cd_english_file", type=str, default="",
                        help="The path to the CD-json file containing the English parts of the CD split")
    parser.add_argument("--language_weight", type=float, default=None,
                        help="weight of contrastive langauge used in the cd files.")
    parser.add_argument("--select_files", type=str, default="3",
                        help="Enter 3 translation files seperated by spaces. (E.g: Path/to/baseline.txt Path/to/CD_0.5.txt Path/to/CD0.9.txt")
    parser.add_argument("--sentences", type=str, nargs="+",
                        help="Optional argument, enter (space seperated) list of integers to represent the sentences you want to look at specifically.")
    args = parser.parse_args()
    main(args)