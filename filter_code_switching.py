import fasttext
from huggingface_hub import hf_hub_download
with open("out/flores/en-de/topk3_final/contrastive-None--0.1-final-topk3-lang-en--0.5.en-de.txt", "r") as CD_05:
    CD_05_sent = CD_05.readlines()
with open("out/flores/en-de/topk3_final/contrastive-None--0.1-final-topk3-lang-en--0.9.en-de.txt", "r") as CD_09:
    CD_09_sent = CD_09.readlines()
with open("out/flores/en-de/topk3_final/final-baseline-topk3.en-de.txt", "r") as base:
    base_sent = base.readlines()
with open("de.txt", "r") as ref:
    ref_sent = ref.readlines()

model2 = fasttext.load_model(hf_hub_download("laurievb/OpenLID", "model.bin"))
#model1 = fasttext.load_model(hf_hub_download("facebook/fasttext-language-identification", "model.bin"))
off_targ_base = []
off_targ_05 = []
off_targ_09 = []
off_targ_ref = []
for idx, trio in enumerate(zip(base_sent, CD_05_sent, CD_09_sent, ref_sent)):
    trio = [x.strip() for x in trio]
    output = model2.predict(trio)

    if output[0][3][0] == "__label__eng_Latn" or output[0][3][0] != "__label__deu_Latn":
        print("Considered English or Non-German")
        off_targ_ref.append(idx)

    if output[0][3][0] == "__label__deu_Latn":
        if output[1][3][0] < 0.57:
            print(trio)
            print(output[1][3][0])

            off_targ_ref.append(idx)


    if output[0][0][0] == "__label__eng_Latn" or output[0][0][0] != "__label__deu_Latn":
        print("Considered English or Non-German")
        off_targ_base.append(idx)

    if output[0][0][0] == "__label__deu_Latn":
        if output[1][0][0] < 0.57:
            #print(trio)
            #print(output[1][0][0])

            off_targ_base.append(idx)

    if output[0][1][0] == "__label__eng_Latn" or output[0][1][0] != "__label__deu_Latn":
        print("Considered English or Non-German")
        off_targ_05.append(idx)

    if output[0][1][0] == "__label__deu_Latn":
        if output[1][1][0] < 0.57:
            #print(trio)
            #print(output[1][1][0])
            off_targ_05.append(idx)

    if output[0][2][0] == "__label__eng_Latn" or output[0][2][0] != "__label__deu_Latn":
        print("Considered English or Non-German")
        off_targ_09.append(idx)

    if output[0][2][0] == "__label__deu_Latn":
        if output[1][2][0] < 0.57:
            #print(trio)
            #print(output[1][2][0])

            off_targ_09.append(idx)


print(off_targ_base, len(off_targ_base))
print(off_targ_05, len(off_targ_05))
print(off_targ_09, len(off_targ_09))
# The threshold for German prediction should be set so that this list of Off-Target references returns empty: e.g.
print("Should be 0", off_targ_ref)

all_3_fails = set(off_targ_base).intersection(off_targ_05).intersection(off_targ_09)
only_base_fails = set(off_targ_base) - set(off_targ_05) - set(off_targ_09)
only_05_fails = set(off_targ_05) - set(off_targ_base) - set(off_targ_09)
only_09_fails = set(off_targ_09) - set(off_targ_base) - set(off_targ_05)

common_base_05 = set(off_targ_base).intersection(off_targ_05) - set(off_targ_09)
common_base_09 = set(off_targ_base).intersection(off_targ_09) - set(off_targ_05)
common_05_09 = set(off_targ_05).intersection(off_targ_09) - set(off_targ_base)

all_3_fails = sorted(list(all_3_fails))
common_base_05 = sorted(list(common_base_05))
common_base_09 = sorted(list(common_base_09))
common_05_09 = sorted(list(common_05_09))
only_base_fails = sorted(list(only_base_fails))
only_05_fails = sorted(list(only_05_fails))
only_09_fails = sorted(list(only_09_fails))
print(f"Common Sentences that fail the Language Identification Test in all 3 Settings: {' '.join(map(str, all_3_fails))},"
      f"\nCommon Sentences that fail the Language Identification Test in Baseline and 0.5 CD: {' '.join(map(str, common_base_05))},"
      f"\nCommon Sentences that fail the Language Identification Test in Baseline and 0.9 CD: {' '.join(map(str, common_base_09))},"
      f"\nCommon Sentences that fail the Language Identification Test in 0.5 and 0.9 CD: {' '.join(map(str, common_05_09))},"
      f"\nOnly Sentences that fail the Language Identification Test in Baseline: {' '.join(map(str, only_base_fails))},"
      f"\nOnly Sentences that fail the Language Identification Test in 0.5 CD: {' '.join(map(str, only_05_fails))},"
      f"\nOnlySentences that fail the Language Identification Test in 0.9 CD: {' '.join(map(str, only_09_fails))},"
      f"\nOff Target Detected from Reference file: {off_targ_ref}")
print(len(all_3_fails) +len(common_base_05) + len(common_base_09) + len(common_05_09) +len(only_base_fails) + len(only_05_fails) +len(only_09_fails))
print(len(all_3_fails) ,len(common_base_05) ,len(common_base_09) , len(common_05_09) ,len(only_base_fails) , len(only_05_fails) ,len(only_09_fails))


