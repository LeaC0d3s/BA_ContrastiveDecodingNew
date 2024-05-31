# Bachelor Thesis Repository
## Off-Target Translations in Large Language Models: A Deep Dive into Language-Contrastive Decoding Outputs and Generated Token Probabilities
Author: Lea Müller


[![made-with-python](https://img.shields.io/badge/Made%20with-Python-red.svg)](#python)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

This repository is for my bachelor thesis. It contains all the code I used for reproducing the outputs for my evaluations as well as all the scrips I wrote in addition to process the obtained data. This README provides an overview of all the files and folder and how to use them.
This repository was initially forked from the Repository "[ContraDecode](https://github.com/ZurichNLP/ContraDecode)" and adapted for my own research goals.

Source-contrastive and language-contrastive decoding, as described in [Sennrich et al. (EACL 2024)](https://arxiv.org/abs/2309.07098):

- In **source-contrastive decoding**, we search for a translation that maximizes P(_Y_|_X_) - λ·P(_Y_|_X'_), where _X'_ is a random source segment. This penalizes hallucinations.

- In **language-contrastive decoding**, we search for a translation that maximizes P(_Y_|_X_,_l_y_) - λ·P(_Y_|_X_,_l_y'_), where _l_y_ is the language indicator for the desired target language, _l_y'_ the indicator for some undesired language (such as English or the source language). This penalizes off-target translations.

<p align="center">
<img src="illustration.png" alt="Our decoding objective yields a translation that is probable given the actual input, but improbable given a source-contrastive or language-contrastive input." width="400">
</p>

For my thesis I only used the **language-contrastive decoding** on the LLM part of their implementation.


## Installation

!Note: The `transformers`library was upgraded from the original version `==4.33.1` to `==4.38.2` as there were some conflicts with my working environement.
- `pip install -r requirements.txt

## Pre-Downloaded Source and Reference Sentences for EN-DE:

This repository automatically opens and uses the pre-downloaded ```devtest``` sentences for English and German from the [FLORES-101](https://huggingface.co/datasets/gsarti/flores_101) dataset when the above command line arguments are run.
- ```de.txt``` The file containing the German references
- ```en.txt``` The file containing the English source sentences

### **Commands used for Reproducing the Outputs and Extracting Probabilities**
 
Baseline with [Llama 2 Chat (13B)](https://arxiv.org/abs/2307.09288) on English–German, using prompting with a one-shot example:
- `python -m scripts.run --model_path llama-2-13b-chat --language_pairs en-de  --oneshot`

Language-contrastive decoding with [Llama 2 Chat (13B)](https://arxiv.org/abs/2307.09288) on English–German, with λ_lang=0.5 and English as contrastive target language, using prompting with a one-shot example:
- `python -m scripts.run --model_path llama-2-13b-chat --language_pairs en-de --language_contrastive en  --language_weight -0.5 --oneshot`

Language-contrastive decoding with [Llama 2 Chat (13B)](https://arxiv.org/abs/2307.09288) on English–German, with λ_lang=0.9 and English as contrastive target language, using prompting with a one-shot example:
- `python -m scripts.run --model_path llama-2-13b-chat --language_pairs en-de --language_contrastive en  --language_weight -0.9 --oneshot`

In addition to the .txt files containing the translations, this repository also produces 3 json files, containing:
- A dictionary with the top 3 most probable tokens at each generation step during decoding (Baseline, λ_lang=0.5 and λ_lang=0.9).
  - out/flores/en-de/topk3_final/final-baseline-topk3.en-de.json
  - out/flores/en-de/topk3_final/contrastive-None--0.1-final-topk3-lang-en--0.5.probs_CD.json
  - out/flores/en-de/topk3_final/contrastive-None--0.1-final-topk3-lang-en--0.9.probs_CD.json
- A dictionary with the top 3 most probable German tokens given the previous generated CD tokens at each step of decoding (λ_lang=0.5 and λ_lang=0.9).
  - out/flores/en-de/topk3_final/contrastive-None--0.1-final-topk3-lang-en--0.5.probs_de_with_fixed_incremental_cd.json
  - out/flores/en-de/topk3_final/contrastive-None--0.1-final-topk3-lang-en--0.9.probs_de_with_fixed_incremental_cd.json
- A dictionary with the top 3 most probable English tokens given the previous generated CD tokens at each step of decoding (λ_lang=0.5 and λ_lang=0.9).
  - out/flores/en-de/topk3_final/contrastive-None--0.1-final-topk3-lang-en--0.5.probs_en_with_fixed_incremental_cd.json
  - out/flores/en-de/topk3_final/contrastive-None--0.1-final-topk3-lang-en--0.9.probs_en_with_fixed_incremental_cd.json


## Code Adaptions from Original Repository
### **mt_task.py**:
1. Local access of source and reference data, instead of directly accessing the datasets via hugging face.
2. Adjusting the return variable of the `evaluate` function, including the dictionary with the probabilities for both Baseline and CD translation.
3. writing the translations and the probability dictionaries into output files.


### **translation_models/\_\_init\_\_.py**: 
- rewriting an assert statement for the `translate` function, as I changed the format of the output to accomodate the probability dictionary.
### **translation_models/llama.py**: 
- functions I adapted for additional probability output:
  - `_translate`,
  - `_translate_multi_source`
- Functions I added to extract the probabilities:
  - `generate_step_by_step`--> calls the German/English prompted model to generate the next token based on the previous tokens produced during CD,
  - `get_runner_ups`--> There are different variations of this code block through multiple functions to accomodate different input formats, but in general they all work similar to this when extracting the second and third most probable token at each generation step, using the normalized scores for probabilities over all possible tokens.


## Additional Scripts for Processing Translations and Probabilities
Here is a list of scripts I put together to process my obtained data:

`compare_files.py`:
  - This script compares the original translation output with the reproduced translations and return all the affected lines, including a total count of the lines that are different.
     - The original produced output files by [Sennrich et al. (EACL 2024)](https://arxiv.org/abs/2309.07098) are saved in: **predictions/llama_flores/en-de/**
     - The reproduced output files are saved in: **out/flores/en-de/topk3_final/**


`compare_probs.py`:
  - This scribt takes two different comman line arguments, where one retuns table containing the Baseline, λ_lang=0.5 and λ_lang=0.9 top 3 probabilities:
      - `python3 -m compare_probs --base_file out/flores/en-de/topk3_final/final-baseline-topk3.en-de.json --cd05_file out/flores/en-de/topk3_final/contrastive-None--0.1-final-topk3-lang-en--0.5.probs_CD.json --cd09_file out/flores/en-de/topk3_final/contrastive-None--0.1-final-topk3-lang-en--0.9.probs_CD.json --sentences 90 281 490 533 679 863 873 875 --filter A`
          - If `--sentences` are specified, you can add the `--filter` argument to name the selected sentences you want to save in a table. Otherwise, it will take the whole dictionary with all 1012 sentences as input.
          - Here you can find the excel sheet, containing the table all the transaltions: --> **Table_Processing/all_prob_comp_table_None.xlsx**
          - Here you can find the excel sheet, containing a collection of Filter A-G Sentence indexes: --> **Table_Processing/prob_comp_table_full.xlsx**
       

  - The second command line argument retuns tables containing a CD output, combined with the German and English token probabilities before CD is applied:
      - ` python3 compare_probs.py --cd_file out/flores/en-de/topk3_final/contrastive-None--0.1-final-topk3-lang-en--0.9.probs_CD.json --cd_german_file out/flores/en-de/topk3_final/contrastive-None--0.1-final-topk3-lang-en--0.9.probs_de_with_fixed_incremental_cd.json --cd_english_file out/flores/en-de/topk3_final/contrastive-No
ne--0.1-final-topk3-lang-en--0.9.probs_en_with_fixed_incremental_cd.json --sentences 90 281 490 533 679 863 873 875 --filter A --language_weight 0.9`
          - Same as the first command, if sentences are specified, you can name them, if no name is given it will return "None". Additionally, you can specify the weight, for the naming of the produced tables.
          - Here you can find the excel sheet, containing the table for λ_lang=0.5 and their split up German and English probabilities for all sentence indexes: --> **Table_Processing/all_prob0.5_comp_table_None.xlsx**
          - Here you can find the excel sheet, containing a collection of Filter A-G Sentence indexes of the same thing: **Table_Processing/prob0.5_comp_table_all_final.xlsx**
       
          - Here you can find the excel sheet, containing the table for λ_lang=0.9 and their split up German and English probabilities for all sentence indexes: --> **Table_Processing/all_prob0.9_comp_table_None.xlsx**
          - Here you can find the excel sheet, containing a collection of Filter A-G Sentence indexes of the same thing: **Table_Processing/prob0.9_comp_table_all_final.xlsx**


## Processing Table Data for Evaluation: Plotting Charts

Here is a short description of the scripts used for processing the obtained probabilities:

`Table_Processing/process_data.py`:
-  This script is used for producing the individual Line Charts in the "Table_Processing/plots" as well as collecting the all indexes from Filter A-G in seperate files, saved in **Table_Processing/Base_05_09_Charts**. The datapoints in this chart that have a token probability below 90% are explicitly labled.

`Table_Processing/process_data_cd05.py`:
-  This script is used for producing the individual Line Charts for a λ_lang=0.5 translation including a scatter plot showing the top 3 most probable token probabilities of German and English at each step, see **Table_Processing/plots_cd05**. Tthe German and English datapoints are only labled when they produce the same value as the top token in the CD top choice.

`Table_Processing/process_data_cd09.py`:
-  This script is used for producing the individual Line Charts for a λ_lang=0.9 translation including a scatter plot showing the top 3 most probable token probabilities of German and English at each step, see **Table_Processing/plots_cd09**. The German and English datapoints are only explicitly labled when they produce the same value as the top token in the CD top choice.

-  In folder **Table_Processing/CD_05_09_Charts** I saved the plotted output, sorted in the Filter A-G, with λ_lang=0.5 and λ_lang=0.9 plots in direct comparison.

`Table_Processing/average_prob_trend.py`:
- This script takes the average token probability of a specific set of translations, containing some type of off-target error. Sentence length of the translations get normalized and produce together with the Baseline, λ_lang=0.5 and λ_lang=0.9 a trend chart. This is repeated for Non Translation, Partial Translation, translation with the remaining error types (excluding: Other) and once for the tanslations that contained no assigned off-target translation error from the extracted indexes during the language identification filtering.
- The Plotted trend Charts are in folder **Table_Processing/Error_Trends**.


## Evaluation of Reproduced Translation (same as [Sennrich et al. (EACL 2024)](https://arxiv.org/abs/2309.07098) )

ChrF2:
```
sacrebleu ref.txt < output.txt --metrics chrf
```


spBLEU:
```
sacrebleu ref.txt < output.txt --tokenize flores101
```


## Reference

```bibtex
@inproceedings{sennrich-etal-2024-mitigating,
      title={Mitigating Hallucinations and Off-target Machine Translation with Source-Contrastive and Language-Contrastive Decoding}, 
      author={Rico Sennrich and Jannis Vamvas and Alireza Mohammadshahi},
      booktitle={18th Conference of the European Chapter of the Association for Computational Linguistics},
      year={2024}
}
```
