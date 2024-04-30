import json

# Open the JSON file
with open('out/flores/en-de/contrastive-None--0.1-lang-en--0.9.probs_orig_de.json', 'r') as f:
    # Load the JSON data
    data_de = json.load(f)
with open('out/flores/en-de/contrastive-None--0.1-lang-en--0.9.probs_orig_en.json', 'r') as f:
    # Load the JSON data
    data_en = json.load(f)
with open('out/flores/en-de/contrastive-None--0.1-lang-en--0.9.probs_CD.json', 'r') as f:
    # Load the JSON data
    data_CD = json.load(f)

with open('out/flores/en-de/contrastive-None--0.1-lang-en--0.9.probs_en_with_fixed_incremental_cd.json', 'r') as f:
    # Load the JSON data
    data_fixed_en = json.load(f)
with open('out/flores/en-de/contrastive-None--0.1-lang-en--0.9.probs_de_with_fixed_incremental_cd.json', 'r') as f:
    # Load the JSON data
    data_fixed_de = json.load(f)

# Now 'data' contains the contents of the JSON file
key_2_sent_de = data_de.get('2')[0]
key_2_sent_en = data_en.get('2')[0]
key_2_sent_CD = data_CD.get('2')[0]
key_2_sent_fixed_en = data_fixed_en.get('2')[1][0][0] #The fixed token
print(key_2_sent_fixed_en)
key_2_sent_fixed_en = data_fixed_en.get('2')[1][0] #All generated tokens with first fixed token
print(key_2_sent_fixed_en)
key_2_sent_fixed_en = data_fixed_en.get('2')[1][0][1] # first translation
print(key_2_sent_fixed_en)
key_2_sent_fixed_en = data_fixed_en.get('2')[1][1][1] # second translation
print(key_2_sent_fixed_en)
key_2_sent_fixed_en = data_fixed_en.get('2')[1][0][1][0]
print(key_2_sent_fixed_en)

key_2_list_de = data_de.get('2')[1]
key_2_list_en = data_en.get('2')[1]
key_2_list_CD = data_CD.get('2')[1]


print("Translate to German - English (0.9): ", key_2_sent_CD)
for idx, token in enumerate(key_2_list_CD):
    if idx == 0:
        print(f"| {token[0]:5d}| {token[1]:9s} | {token[2]:.4f} | {token[3]} | ")
    else:
        print(f"| {token[0]:5d}| {token[1]:9s} | {token[2]:.4f} | {token[3]} | To Eng: {data_fixed_en.get('2')[1][idx-1][1][0]} | To Ger: {data_fixed_de.get('2')[1][idx-1][1][0]}")
print("Translate to German: ", key_2_sent_de)
for token in key_2_list_de:
    print(f"| {token[0]:5d}| {token[1]:9s} | {token[2]:.4f} | {token[3]} |")
print("Translate to English: ", key_2_sent_en)
for token in key_2_list_en:
    print(f"| {token[0]:5d}| {token[1]:9s} | {token[2]:.4f} | {token[3]} |")
![](../Desktop/Output_Probabilities_for_CD_0.9_with_en_de.png)


