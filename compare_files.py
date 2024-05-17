file1_path = "predictions/llama_flores/en-de/Llama-2-13b-chat-1shot-baseline.de"
file1_path = "predictions/llama_flores/en-de/Llama-2-13b-chat-1shot-contrastive-en-0.5.de"
file1_path = "predictions/llama_flores/en-de/Llama-2-13b-chat-1shot-contrastive-en-0.9.de"
file2_path = "out/flores/en-de/direct-control-NT.en-de.txt"
file2_path = "out/flores/en-de/contrastive-None--0.1-topk3-lang-en--0.5.en-de.txt"
file2_path = "out/flores/en-de/contrastive-None--0.1-topk3-lang-en--0.9.en-de.txt"

count=0
with open(file1_path, "r") as file1, open(file2_path, "r") as file2:
    for line_number, (line1, line2) in enumerate(zip(file1, file2), start=1):
        if line1.strip() != line2.strip():
            count+=1
            print(f"Line {line_number}:")
            print(f"File1: original pred: {line1.strip()}")
            print(f"File2: reproduced out: {line2.strip()}")
            print()
print(count)
