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


def process_sheet_05_all(sheet_name, excel_file, output_path):
    # Read the sheet
    df = pd.read_excel(excel_file, sheet_name=sheet_name, header=2)

    df.columns = [
        'Idx', 'tok1_int_cd5', 'tok1_cd5', 'tok1_cd5%', 'tok2_int_cd5', 'tok2_cd5', 'tok2_cd5%', 'tok3_int_cd5',
        'tok3_cd5', 'tok3_cd5%',
        'tok1_int_ger', 'tok1_ger', 'tok1_ger%', 'tok2_int_ger', 'tok2_ger', 'tok2_ger%', 'tok3_int_ger', 'tok3_ger',
        'tok3_ger%',
        'tok1_int_en', 'tok1_en', 'tok1_en%', 'tok2_int_en', 'tok2_en', 'tok2_en%', 'tok3_int_en', 'tok3_en', 'tok3_en%'
    ]
    # Filter rows where 'tok1_cd5' is the same as 'tok1_ger' but different from 'tok1_en'
    filtered_df = df[
        (df['tok1_int_cd5'] == df['tok1_int_ger']) &
        (df['tok1_int_cd5'] == df['tok1_int_en'])
        ]
    filtered_df_new = df[
        (df['tok1_int_cd5'] != df['tok1_int_ger'])
        ]
    # 13 is the token integer for \n characters.
    # \n indicates the end or start of a translation, therefore we remove it from the calculations:
    filtered_df = filtered_df[filtered_df['tok1_int_cd5'] != 13]
    filtered_df_new = filtered_df_new[filtered_df_new['tok1_int_cd5'] != 13]
    filtered_df_new.to_excel(output_path, sheet_name=sheet_name, index=False)

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


def process_sheet_comp_all(sheet_name, excel_file, output_path):
    df = pd.read_excel(excel_file, sheet_name=sheet_name, header=2)
    df.columns = [
        'Idx', 'tok1_int_b', 'tok1_b', 'tok1_b%', 'tok2_int_b', 'tok2_b', 'tok2_b%', 'tok3_int_b',
        'tok3_b', 'tok3_b%',
        'tok1_int_cd5', 'tok1_cd5', 'tok1_cd5%', 'tok2_int_cd5', 'tok2_cd5', 'tok2_cd5%', 'tok3_int_cd5',
        'tok3_cd5', 'tok3_cd5%',
        'tok1_int_cd9', 'tok1_cd9', 'tok1_cd9%', 'tok2_int_cd9', 'tok2_cd9', 'tok2_cd9%', 'tok3_int_cd9',
        'tok3_cd9', 'tok3_cd9%'
    ]
    filtered_df = df[
        (df['tok1_int_b'] == df['tok1_int_cd5']) &
        (df['tok1_int_b'] == df['tok1_int_cd9'])
        ]
    filtered_df = filtered_df[filtered_df['tok1_int_b'] != 13]

    if not filtered_df.empty:
        # Convert to numeric and calculate 'Prob_diff'
        filtered_df['tok1_b%'] = pd.to_numeric(filtered_df['tok1_b%'], errors='coerce')
        filtered_df['tok1_cd5%'] = pd.to_numeric(filtered_df['tok1_cd5%'], errors='coerce')
        filtered_df['tok1_cd9%'] = pd.to_numeric(filtered_df['tok1_cd9%'], errors='coerce')

        filtered_df['Prob_diff_b_5'] = filtered_df['tok1_b%'] - filtered_df['tok1_cd5%']
        filtered_df['Prob_diff_b_9'] = filtered_df['tok1_b%'] - filtered_df['tok1_cd9%']


        # Calculate mean and standard deviation
        prob_diff_mean = np.mean(filtered_df['Prob_diff_b_5'])
        prob_diff_std = np.std(filtered_df['Prob_diff_b_5'])

        prob_diff_mean = np.mean(filtered_df['Prob_diff_b_9'])
        prob_diff_std = np.std(filtered_df['Prob_diff_b_9'])

        return prob_diff_mean, prob_diff_std
    else:
        return None, None


def process_sheet_09_all(sheet_name, excel_file, output_path):
    # Read the sheet
    df = pd.read_excel(excel_file, sheet_name=sheet_name, header=2)

    df.columns = [
        'Idx', 'tok1_int_cd9', 'tok1_cd9', 'tok1_cd9%', 'tok2_int_cd9', 'tok2_cd9', 'tok2_cd9%', 'tok3_int_cd9',
        'tok3_cd9', 'tok3_cd9%',
        'tok1_int_ger', 'tok1_ger', 'tok1_ger%', 'tok2_int_ger', 'tok2_ger', 'tok2_ger%', 'tok3_int_ger', 'tok3_ger',
        'tok3_ger%',
        'tok1_int_en', 'tok1_en', 'tok1_en%', 'tok2_int_en', 'tok2_en', 'tok2_en%', 'tok3_int_en', 'tok3_en', 'tok3_en%'
    ]
    # Filter rows where 'tok1_cd5' is the same as 'tok1_ger' but different from 'tok1_en'
    filtered_df = df[
        (df['tok1_int_cd9'] == df['tok1_int_ger']) &
        (df['tok1_int_cd9'] == df['tok1_int_en'])
        ]

    filtered_df_new = df[
        (df['tok1_int_cd9'] != df['tok1_int_ger'])
    ]
    # 13 is the token integer for \n characters.
    # \n indicates the end or start of a translation, therefore we remove it from the calculations:
    filtered_df = filtered_df[filtered_df['tok1_int_cd9'] != 13]
    filtered_df_new = filtered_df_new[filtered_df_new['tok1_int_cd9'] != 13]

    filtered_df_new.to_excel(output_path, sheet_name=sheet_name, index=False)

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
file_path_comp = 'prob_comp_table_full.xlsx'
excel_data_comp = pd.ExcelFile(file_path_comp)

file_path_05 = 'prob0.5_comp_table_all_final.xlsx'
excel_data_05 = pd.ExcelFile(file_path_05)

file_path_09 = 'prob0.9_comp_table_all_final.xlsx'
excel_data_09 = pd.ExcelFile(file_path_09)

file_path_05_all = 'all_prob0.5_comp_table_None.xlsx'
excel_data_05_all = pd.ExcelFile(file_path_05_all)

file_path_09_all = 'all_prob0.9_comp_table_None.xlsx'
excel_data_09_all = pd.ExcelFile(file_path_09_all)




# Process sheets A-G from the new file with the new condition and collect results
# reduced tables
sheets_09_redu = [
    'Blatt A - prob0.9_comp_table_re', 'Blatt B - prob0.9_comp_table_re',
    'Blatt C - prob0.9_comp_table_re', 'Blatt D - prob0.9_comp_table_re',
    'Blatt E - prob0.9_comp_table_re', 'Blatt F - prob0.9_comp_table_re',
    'Blatt G - prob0.9_comp_table_re'
]
sheets_05_redu = [
    'Blatt A - prob0.5_comp_table_re', 'Blatt B - prob0.5_comp_table_re',
    'Blatt C - prob0.5_comp_table_re', 'Blatt D - prob0.5_comp_table_re',
    'Blatt E - prob0.5_comp_table_re', 'Blatt F - prob0.5_comp_table_re',
    'Blatt G - prob0.5_comp_table_re'
]

# tables incl. token integer
sheets_09 = [
    'Blatt A - prob0.9_comp_table_A', 'Blatt B - prob0.9_comp_table_B',
    'Blatt C - prob0.9_comp_table_C', 'Blatt D - prob0.9_comp_table_D',
    'Blatt E - prob0.9_comp_table_E', 'Blatt F - prob0.9_comp_table_F',
    'Blatt G - prob0.9_comp_table_G'
]
sheets_05 = [
    'Blatt A - prob0.5_comp_table_A', 'Blatt B - prob0.5_comp_table_B',
    'Blatt C - prob0.5_comp_table_C', 'Blatt D - prob0.5_comp_table_D',
    'Blatt E - prob0.5_comp_table_E', 'Blatt F - prob0.5_comp_table_F',
    'Blatt G - prob0.5_comp_table_G'
]

sheet_comp = ['Blatt A - prob_comp_table_A_ful', 'Blatt B - prob_comp_table_B_ful',
              'Blatt C - prob_comp_table_C_ful', 'Blatt D - prob_comp_table_D_ful',
              'Blatt E - prob_comp_table_E_ful', 'Blatt F - prob_comp_table_F_ful',
              'Blatt G - prob_comp_table_G_ful',
]

sheets_05_all = [
    'Blatt All - all_prob0.5_comp_ta'
]
sheets_09_all = [
    'Blatt All - all_prob0.9_comp_ta'
]

#for reduced tables
"""
results_05_condition_redu = {}
for sheet in sheets_05_redu:
    output_path = f"New_Filtered_tables/filtered_0.5_{sheet}.xlsx"
    mean, std = process_sheet_05(sheet, file_path_05, output_path)
    results_05_condition_redu[sheet] = {'mean': mean, 'std': std}

# for reduced tables
results_09_condition_redu = {}
for sheet in sheets_09_redu:
    output_path = f"New_Filtered_tables/filtered_0.9_{sheet}.xlsx"
    mean, std = process_sheet_09(sheet, file_path_09, output_path)
    results_09_condition_redu[sheet] = {'mean': mean, 'std': std}

"""

results_comp_condition = {}

for sheet in sheet_comp:
    output_path = f'New_Filtered_tables/All_comp_{sheet}.xlsx'
    mean, std = process_sheet_comp_all(sheet, file_path_comp, output_path)


results_05_condition = {}
for sheet in sheets_05:
    output_path = f'New_Filtered_tables/All_0.5_{sheet}.xlsx'
    mean, std = process_sheet_05_all(sheet, file_path_05, output_path)
    results_05_condition[sheet] = {'mean': mean, 'std': std}

results_09_condition = {}
for sheet in sheets_09:
    output_path = f'New_Filtered_tables/All_0.9_{sheet}.xlsx'
    mean, std = process_sheet_09_all(sheet, file_path_09, output_path)
    results_09_condition[sheet] = {'mean': mean, 'std': std}


results_05_condition_all = {}
for sheet in sheets_05_all:
    output_path = f'New_Filtered_tables/All_0.5_{sheet}.xlsx'
    mean, std = process_sheet_05_all(sheet, file_path_05_all, output_path)
    results_05_condition_all[sheet] = {'mean': mean, 'std': std}

results_09_condition_all = {}
for sheet in sheets_09_all:
    output_path = f'New_Filtered_tables/All_0.9_{sheet}.xlsx'
    mean, std = process_sheet_09_all(sheet, file_path_09_all, output_path)
    results_09_condition_all[sheet] = {'mean': mean, 'std': std}


print(results_05_condition)
print(results_09_condition)

print(results_05_condition_all)
print(results_09_condition_all)


