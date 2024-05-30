import pandas as pd
import matplotlib.pyplot as plt
import os
import xlsxwriter
import numpy as np

file_path = 'all_prob_comp_table_None.xlsx'  # Replace with your actual file path
xls = pd.ExcelFile(file_path)

# Read the 'Blatt A - prob_comp_table_A_ful' table from row 2
table = pd.read_excel(xls, 'Blatt 1 - all_prob_comp_table_N', header=2)

table.columns = [
        'Idx', 'tok1_int_b', 'tok1_b', 'tok1_b%', 'tok2_int_b', 'tok2_b', 'tok2_b%', 'tok3_int_b',
        'tok3_b', 'tok3_b%',
        'tok1_int_cd5', 'tok1_cd5', 'tok1_cd5%', 'tok2_int_cd5', 'tok2_cd5', 'tok2_cd5%', 'tok3_int_cd5',
        'tok3_cd5', 'tok3_cd5%',
        'tok1_int_cd9', 'tok1_cd9', 'tok1_cd9%', 'tok2_int_cd9', 'tok2_cd9', 'tok2_cd9%', 'tok3_int_cd9',
        'tok3_cd9', 'tok3_cd9%'
    ]



# Filter the data to include only the specified indices that contain NoT errors
selected_indices_b = [90, 281, 533, 863, 873, 875, 92, 273, 505, 532, 683,824,826,
                    830,853,874, 956, 1, 133, 134, 173, 245, 296,
                    311, 375, 422, 484, 494, 558, 559, 629, 655, 740, 871, 889, 891, 934]
selected_indices_cd5 = [90, 863, 873, 875, 683, 824, 826, 830, 853, 874]
selected_indices_cd9 = [90, 863, 873, 875]
error_type = "NoT"
# Filter the data to include only the specified indices that contain PartT errors
selected_indices_b = [490,679,107,172,474, 615,847,914,154, 196, 754, 877]
selected_indices_cd5 = [490,533,679,107,172,474, 615,847, 914, 12, 327, 974, 217, 444, 702]
selected_indices_cd9 = [490,533,679,956, 12, 327, 974, 121, 210,365,387, 614, 772, 851, 893]
error_type = "PartT"

# Filter the data to include only the specified indices that contain Insert and or NonW errors
selected_indices_b = [467, 444, 702, 772, 893]
selected_indices_cd5 = [281, 467, 154, 196, 558, 559, 629,655, 754,871,877, 889, 934, 772, 893]
selected_indices_cd9 = [281, 92, 107, 172,467,474, 505, 615, 824, 826, 847, 853, 874, 936, 196, 375, 558, 559, 629, 655,754,871,889, 934,444,702, 41, 469]
error_type = "Insert_NonW_mix"

# Filter the data to include only the specified indices that contain no Error Type
selected_indices_b = [12, 327, 974,217, 41, 121, 210, 365, 387, 469, 614, 851]
selected_indices_cd5 = [956, 2, 133, 134, 173, 245,296,311,375, 422, 484, 494, 740, 891, 41, 121, 210, 365, 387, 469, 614, 851]
selected_indices_cd9 = [273, 532, 830, 914,2, 133, 134, 154, 173, 245,296,311,422, 484, 494, 740, 877, 891, 217]
error_type = "No"



filtered_data_b = table[table['tok1_int_b'] != 13].dropna(subset=['tok1_b%'])
filtered_data_cd5 = table[table['tok1_int_cd5'] != 13].dropna(subset=['tok1_cd5%'])
filtered_data_cd9 = table[table['tok1_int_cd9'] != 13].dropna(subset=['tok1_cd9%'])

filtered_data_b = filtered_data_b[filtered_data_b['Idx'].isin(selected_indices_b)]
filtered_data_cd5 = filtered_data_cd5[filtered_data_cd5['Idx'].isin(selected_indices_cd5)]
filtered_data_cd9 = filtered_data_cd9[filtered_data_cd9['Idx'].isin(selected_indices_cd9)]


# Group the filtered data by 'Idx' and extract 'tok1_b%' probabilities
filtered_grouped_data_b = filtered_data_b.groupby('Idx')['tok1_b%'].apply(list)
filtered_grouped_data_cd5 = filtered_data_cd5.groupby('Idx')['tok1_cd5%'].apply(list)
filtered_grouped_data_cd9 = filtered_data_cd9.groupby('Idx')['tok1_cd9%'].apply(list)


# Normalize the length of each sentence to 1
max_length = max(filtered_grouped_data_b.apply(len))
max_length_cd5 = max(filtered_grouped_data_cd5.apply(len))
max_length_cd9 = max(filtered_grouped_data_cd9.apply(len))

# Interpolating and reindexing to normalize lengths
normalized_probabilities = filtered_grouped_data_b.apply(
    lambda x: np.interp(np.linspace(0, len(x) - 1, max_length), np.arange(len(x)), x)
)

normalized_probabilities_cd5 = filtered_grouped_data_cd5.apply(
    lambda x: np.interp(np.linspace(0, len(x) - 1, max_length_cd5), np.arange(len(x)), x)
)

normalized_probabilities_cd9 = filtered_grouped_data_cd9.apply(
    lambda x: np.interp(np.linspace(0, len(x) - 1, max_length_cd9), np.arange(len(x)), x)
)

# Compute the average probability across the selected sentences
average_probabilities = np.mean(normalized_probabilities.tolist(), axis=0)
average_probabilities_cd5 = np.mean(normalized_probabilities_cd5.tolist(), axis=0)
average_probabilities_cd9 = np.mean(normalized_probabilities_cd9.tolist(), axis=0)

output_dir = 'Error_Trends'
os.makedirs(output_dir, exist_ok=True)

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(average_probabilities, label='Average tok1_b%', color="blue")
plt.plot(average_probabilities_cd5, label='Average tok1_cd5%', color= "green")
plt.plot(average_probabilities_cd9, label='Average tok1_cd9%', color="red")
plt.title(f'Average Token Probability Across Normalized Sentences (containing {error_type} errors)')
plt.xlabel('Normalized Sentence Length')
plt.ylabel('Probability')
plt.legend()
plt.grid(True)
plot_file = os.path.join(output_dir, f'plot_{error_type}.png')
plt.savefig(plot_file)
plt.show()
