# Define a function to process each sheet from the new file
import pandas as pd
import numpy as np


# Define a function to process each sheet for the new condition
def process_sheet_05(sheet_name, excel_file, output_path):
    # Read the sheet
    df = pd.read_excel(excel_file, sheet_name=sheet_name, header=2)

    # Rename columns
    df.columns = [
        'Idx', 'tok1_cd5', 'tok1_cd5%', 'tok2_cd5', 'tok2_cd5%', 'tok3_cd5', 'tok3_cd5%',
        'tok1_ger', 'tok1_ger%', 'tok2_ger', 'tok2_ger%', 'tok3_ger', 'tok3_ger%',
        'tok1_en', 'tok1_en%', 'tok2_en', 'tok2_en%', 'tok3_en', 'tok3_en%'
    ]

    # Filter rows where 'tok1_cd5' is the same as 'tok1_ger' but different from 'tok1_en'
    filtered_df = df[
        (df['tok1_cd5'] == df['tok1_ger']) &
        (df['tok1_cd5'] != df['tok1_en'])
        ]
    #remove all rows that generated a \n token during CD:
    #dropped_rows = filtered_df[filtered_df['tok1_cd5'] == '\\n']
    filtered_df = filtered_df[filtered_df['tok1_cd5'] != '\\n']
    filtered_df.to_excel(output_path, sheet_name=sheet_name, index=False)

    if not filtered_df.empty:
        # Convert to numeric and calculate 'Prob_diff'
        filtered_df['tok1_cd5%'] = pd.to_numeric(filtered_df['tok1_cd5%'], errors='coerce')
        filtered_df['tok1_ger%'] = pd.to_numeric(filtered_df['tok1_ger%'], errors='coerce')
        filtered_df['Prob_diff'] = filtered_df['tok1_cd5%'] - filtered_df['tok1_ger%']

        # Calculate mean and standard deviation
        prob_diff_mean = np.mean(filtered_df['Prob_diff'])
        prob_diff_std = np.std(filtered_df['Prob_diff'])

        return prob_diff_mean, prob_diff_std
    else:
        return None, None


def process_sheet_09(sheet_name, excel_file, output_path):
    # Read the sheet
    df = pd.read_excel(excel_file, sheet_name=sheet_name, header=2)

    # Rename columns
    df.columns = [
        'Idx', 'tok1_cd9', 'tok1_cd9%', 'tok2_cd9', 'tok2_cd9%', 'tok3_cd9', 'tok3_cd9%',
        'tok1_ger', 'tok1_ger%', 'tok2_ger', 'tok2_ger%', 'tok3_ger', 'tok3_ger%',
        'tok1_en', 'tok1_en%', 'tok2_en', 'tok2_en%', 'tok3_en', 'tok3_en%'
    ]

    # Filter rows where 'tok1_cd9' is the same as 'tok1_ger' but different from 'tok1_en'
    filtered_df = df[
        (df['tok1_cd9'] == df['tok1_ger']) &
        (df['tok1_cd9'] != df['tok1_en'])
        ]
    # remove all rows that generated a \n token during CD:
    #dropped_rows = filtered_df[filtered_df['tok1_cd5'] == '\\n']
    filtered_df = filtered_df[filtered_df['tok1_cd9'] != '\\n']
    filtered_df.to_excel(output_path, sheet_name=sheet_name, index=False)


    if not filtered_df.empty:
        # Convert to numeric and calculate 'Prob_diff'
        filtered_df['tok1_cd9%'] = pd.to_numeric(filtered_df['tok1_cd9%'], errors='coerce')
        filtered_df['tok1_ger%'] = pd.to_numeric(filtered_df['tok1_ger%'], errors='coerce')
        filtered_df['Prob_diff'] = filtered_df['tok1_cd9%'] - filtered_df['tok1_ger%']

        # Calculate mean and standard deviation
        prob_diff_mean = np.mean(filtered_df['Prob_diff'])
        prob_diff_std = np.std(filtered_df['Prob_diff'])

        return prob_diff_mean, prob_diff_std
    else:
        return None, None


# Load the new Excel file
file_path_05 = 'prob0.5_comp_table_redu_all_new.xlsx'
excel_data_05 = pd.ExcelFile(file_path_05)

file_path_09 = 'prob0.9_comp_table_redu_all_new.xlsx'
excel_data_09 = pd.ExcelFile(file_path_09)


# Process sheets A-G from the new file with the new condition and collect results
sheets_09 = [
    'Blatt A - prob0.9_comp_table_re', 'Blatt B - prob0.9_comp_table_re',
    'Blatt C - prob0.9_comp_table_re', 'Blatt D - prob0.9_comp_table_re',
    'Blatt E - prob0.9_comp_table_re', 'Blatt F - prob0.9_comp_table_re',
    'Blatt G - prob0.9_comp_table_re'
]
sheets_05 = [
    'Blatt A - prob0.5_comp_table_re', 'Blatt B - prob0.5_comp_table_re',
    'Blatt C - prob0.5_comp_table_re', 'Blatt D - prob0.5_comp_table_re',
    'Blatt E - prob0.5_comp_table_re', 'Blatt F - prob0.5_comp_table_re',
    'Blatt G - prob0.5_comp_table_re'
]

results_05_condition = {}
for sheet in sheets_05:
    output_path = f"Filtered_tables/filtered_0.5_{sheet}.xlsx"
    mean, std = process_sheet_05(sheet, file_path_05, output_path)
    results_05_condition[sheet] = {'mean': mean, 'std': std}

results_09_condition = {}
for sheet in sheets_09:
    output_path = f"Filtered_tables/filtered_0.9_{sheet}.xlsx"
    mean, std = process_sheet_09(sheet, file_path_09, output_path)
    results_09_condition[sheet] = {'mean': mean, 'std': std}



print(results_09_condition)
print(results_05_condition)

