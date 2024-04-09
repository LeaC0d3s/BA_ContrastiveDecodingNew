import json

# Open the JSON file
with open('out/flores/en-de/contrastive-None--0.1-lang-en--0.9.probs_orig_en.json', 'r') as f:
    # Load the JSON data
    data = json.load(f)

# Now 'data' contains the contents of the JSON file
key_19_sent = data.get('19')[0]

key_19_list = data.get('19')[1]

print(key_19_sent)
for token in key_19_list:
    print(f"| {token[0]:5d}| {token[1]:9s} | {token[2]:.4f} | {token[3]} |")


