from __future__ import print_function, division
import sys
from collections import defaultdict
from math import log, exp

#maximum length to show
max_size = int(sys.argv[1])

#source sentences
f_src = open(sys.argv[2])
#references
f_ref = open(sys.argv[3])

#hypotheses
fs = sys.argv[4:]

d = defaultdict(dict)

src = [line.strip() for line in f_src.readlines()]
refs = [line.strip() for line in f_ref.readlines()]

print('1', sys.argv[2])
print('2', sys.argv[3])

for i, f in enumerate(fs):
    print(str(i+3), f)
print('')

def extract_ngrams(sentence, max_length=4):
    words = sentence.split()
    results = defaultdict(lambda: defaultdict(int))
    for length in range(max_length):
        for start_pos in range(len(words)):
            end_pos = start_pos + length + 1
            if end_pos <= len(words):
                results[length][tuple(words[start_pos: end_pos])] += 1
    return results


def score_bleu(ngrams_ref, ngrams_test, correct, total):

    for rank in ngrams_test:
        for chain in ngrams_test[rank]:
            total[rank] += ngrams_test[rank][chain]
            if chain in ngrams_ref[rank]:
                correct[rank] += min(ngrams_test[rank][chain], ngrams_ref[rank][chain])

    return correct, total


def calc(correct, total_hyp, ref_len, smooth=0, max_length=4):

    logbleu = 0

    for i in range(max_length):
      if total_hyp[i] + smooth:
        logbleu += log((correct[i] + smooth) / (total_hyp[i] + smooth))

    logbleu /= max_length

    brevity = 1.0 - ref_len / max(1,total_hyp[0]);

    if brevity < 0.0:
      logbleu += brevity

    return exp(logbleu)

def comparison(key, d1, d2):
    bleu1 = d1[key][0]
    bleu2 = d2[key][0]
    ref = refs[key]
    #return bleu1 # return output based on the bleu score of the first translation
    return bleu1-bleu2 # return output based on difference between bleu scores

for i,f in enumerate(fs):
    f = open(f)
    for j, line in enumerate(f):
        hypo = line.strip()
        ngrams_ref = extract_ngrams(refs[j])
        ref_len = sum(ngrams_ref[0].values())
        correct = [0]*4
        total = [0]*4
        ngrams_test = extract_ngrams(hypo)
        score_bleu(ngrams_ref, ngrams_test, correct, total)
        bleu = calc(correct, total, ref_len, smooth=1)
        d[i][j] = (bleu, hypo)

#Lists of Prefiltered Sentences: Filter A - Filter G
#fa = [90, 308, 310, 490, 533, 679, 863, 873, 875]
#fb = [42, 85, 92, 107, 172, 273, 467, 474, 505, 532, 615, 683, 824, 826, 830, 837, 847, 853, 874, 914, 924, 936]
#fc = [956]
#fd = [12, 22, 327, 974]
#fe = [2, 3, 79, 133, 134, 154, 173, 196, 245, 268, 281, 296, 311, 375, 422, 484, 494, 558, 559, 629, 655, 740, 754, 871, 877, 889, 891, 934]
#ff = [217, 444, 702]
#fg = [41, 121, 192, 210, 260, 270, 306, 365, 387, 450, 469, 550, 614, 772, 851, 893, 986]


worse = sorted(d[0], key= lambda x: comparison(x,d[0],d[1]), reverse=True)
# Replace "worse" with a custom list of indexes.
for entry in worse:
    if len(refs[entry].split()) > max_size:
        continue
    print(entry)
    print(' '.join([str(d[i][entry][0]) for i in range(len(fs))]))
    print('1', src[entry])
    print('2', refs[entry])
    for i in range(len(fs)):
        print(str(i+3), d[i][entry][1])
    print('')


