import pandas as pd
import matplotlib.pyplot as plt
import os
import xlsxwriter

# Load the Excel file
file_path = 'prob0.5_comp_table_all_final.xlsx'  # Replace with your actual file path
xls = pd.ExcelFile(file_path)

# Read the 'Blatt A - prob_comp_table_A_ful' table from row 2 (Replace A with B-G to access the other tables
table_a = pd.read_excel(xls, 'Blatt G - prob0.5_comp_table_G', header=2)

table_a.columns = [
        'Idx', 'tok1_int_cd5', 'tok1_cd5', 'tok1_cd5%', 'tok2_int_cd5', 'tok2_cd5', 'tok2_cd5%', 'tok3_int_cd5',
        'tok3_cd5', 'tok3_cd5%',
        'tok1_int_ger', 'tok1_ger', 'tok1_ger%', 'tok2_int_ger', 'tok2_ger', 'tok2_ger%', 'tok3_int_ger', 'tok3_ger',
        'tok3_ger%',
        'tok1_int_en', 'tok1_en', 'tok1_en%', 'tok2_int_en', 'tok2_en', 'tok2_en%', 'tok3_int_en', 'tok3_en', 'tok3_en%'
    ]

# Display the first few rows of table A to verify the structure
print("Initial table structure:\n", table_a.columns)

# Filter out rows where tok1_int_cd5 = 13 and NaN rows
filtered_table_a = table_a[table_a['tok1_int_cd5']!= 13].dropna(subset=['tok1_int_ger'])

# Group by 'Idx'
grouped = filtered_table_a.groupby('Idx')

filtered_table_a['tok1_cd5%'] = pd.to_numeric(filtered_table_a['tok1_cd5%'], errors='coerce')
filtered_table_a['tok1_ger%'] = pd.to_numeric(filtered_table_a['tok1_ger%'], errors='coerce')
filtered_table_a['tok2_ger%'] = pd.to_numeric(filtered_table_a['tok2_ger%'], errors='coerce')
filtered_table_a['tok3_ger%'] = pd.to_numeric(filtered_table_a['tok3_ger%'], errors='coerce')
filtered_table_a['tok1_en%'] = pd.to_numeric(filtered_table_a['tok1_en%'], errors='coerce')
filtered_table_a['tok2_en%'] = pd.to_numeric(filtered_table_a['tok2_en%'], errors='coerce')
filtered_table_a['tok3_en%'] = pd.to_numeric(filtered_table_a['tok3_en%'], errors='coerce')




# Create a directory to save the plots
output_dir = 'plots_cd05'
os.makedirs(output_dir, exist_ok=True)

plot_files = []
# For each group, create a diagram showing the probabilities
for name, group in grouped:
    #print(name, group)
    plt.figure(figsize=(12, 6))
    max_length = len(group)


    if 'tok1_cd5%' in group.columns and not group['tok1_cd5%'].isna().all():
        tok1_cd5_percents = group['tok1_cd5%'].reset_index(drop=True)
        plt.plot(range(1, len(tok1_cd5_percents) + 1), tok1_cd5_percents, label='tok1_cd5%', marker='o', color='green', zorder=1)

    for i in range(len(group)):
        if pd.notna(group['tok1_ger%'].iloc[i]):
            plt.scatter(i + 1, group['tok1_ger%'].iloc[i], label='tok1_ger%' if i == 0 else "", color='#3C3D99',marker='*',zorder=6)
            if group['tok1_int_cd5'].iloc[i] == group['tok1_int_ger'].iloc[i]:
                plt.annotate(group['tok1_ger'].iloc[i], (i + 1, group['tok1_ger%'].iloc[i]), ha='center', fontsize=8,
                             xytext=(0, -10), textcoords='offset points')

        if pd.notna(group['tok2_ger%'].iloc[i]):
            plt.scatter(i + 1, group['tok2_ger%'].iloc[i], label='tok2_ger%' if i == 0 else "", color='#8481DD',marker='*',zorder=5)
            if group['tok1_int_cd5'].iloc[i] == group['tok2_int_ger'].iloc[i]:
                plt.annotate(group['tok2_ger'].iloc[i], (i + 1, group['tok2_ger%'].iloc[i]), ha='center', fontsize=8,
                             xytext=(0, -10), textcoords='offset points')

        if pd.notna(group['tok3_ger%'].iloc[i]):
            plt.scatter(i + 1, group['tok3_ger%'].iloc[i], label='tok3_ger%' if i == 0 else "", color='#B2B0EA', marker='*',zorder=4)
            if group['tok1_int_cd5'].iloc[i] == group['tok3_int_ger'].iloc[i]:
                plt.annotate(group['tok3_ger'].iloc[i], (i + 1, group['tok3_ger%'].iloc[i]), ha='center', fontsize=8,
                             xytext=(0, -10), textcoords='offset points')


    for i in range(len(group)):
        if pd.notna(group['tok1_en%'].iloc[i]):
            plt.scatter(i + 1, group['tok1_en%'].iloc[i], label='tok1_en%' if i == 0 else "", color='#8F4700', marker='s',zorder=3)
            if group['tok1_int_cd5'].iloc[i] == group['tok1_int_en'].iloc[i]:
                plt.annotate(group['tok1_en'].iloc[i], (i + 1, group['tok1_en%'].iloc[i]), ha='center', fontsize=8,
                             xytext=(0, -10), textcoords='offset points')

        if pd.notna(group['tok2_en%'].iloc[i]):
            plt.scatter(i + 1, group['tok2_en%'].iloc[i], label='tok2_en%' if i == 0 else "", color='#C46100', marker='s',zorder=2)
            if group['tok1_int_cd5'].iloc[i] == group['tok2_int_en'].iloc[i]:
                plt.annotate(group['tok2_en'].iloc[i], (i + 1, group['tok2_en%'].iloc[i]), ha='center', fontsize=8,
                         xytext=(0, -10), textcoords='offset points')
        if pd.notna(group['tok3_en%'].iloc[i]):
            plt.scatter(i + 1, group['tok3_en%'].iloc[i], label='tok3_en%' if i == 0 else "", color='#EC7A08', marker='s',zorder=2)
            if group['tok1_int_cd5'].iloc[i] == group['tok3_int_en'].iloc[i]:
                plt.annotate(group['tok3_en'].iloc[i], (i + 1, group['tok3_en%'].iloc[i]), ha='center', fontsize=8,
                         xytext=(0, -10), textcoords='offset points')

    """
    if 'tok1_b%' in group.columns:
        tok1_b_percents = group['tok1_b%'].reset_index(drop=True)
        max_length = max(max_length, len(tok1_b_percents))

        plt.plot(range(1, len(tok1_b_percents) + 1), tok1_b_percents, label='tok1_b%', marker='o')
        for i, txt in enumerate(group['tok1_b'].reset_index(drop=True)):
            if tok1_b_percents.iloc[i] < 0.9:  # Convert 0.8 to percentage for comparison
                plt.annotate(txt, (i + 1, tok1_b_percents.iloc[i]), xytext=(0, -10), textcoords='offset points', ha='center')

    if 'tok1_cd5%' in group.columns:
        tok1_cd5_percents = group['tok1_cd5%'].reset_index(drop=True)
        max_length = max(max_length, len(tok1_cd5_percents))

        plt.plot(range(1, len(tok1_cd5_percents) + 1), tok1_cd5_percents, label='tok1_cd5%', marker='o')
        for i, txt in enumerate(group['tok1_cd5'].reset_index(drop=True)):
            if tok1_cd5_percents.iloc[i] < 0.9:  # Convert 0.8 to percentage for comparison
                plt.annotate(txt, (i + 1, tok1_cd5_percents.iloc[i]), xytext=(0, -10), textcoords='offset points', ha='center')

    if 'tok1_cd9%' in group.columns:
        tok1_cd9_percents = group['tok1_cd9%'].reset_index(drop=True)
        max_length = max(max_length, len(tok1_cd9_percents))

        plt.plot(range(1, len(tok1_cd9_percents) + 1), tok1_cd9_percents, label='tok1_cd9%', marker='o')
        for i, txt in enumerate(group['tok1_cd9'].reset_index(drop=True)):
            if tok1_cd9_percents.iloc[i] < 0.9:  # Convert 0.8 to percentage for comparison
                plt.annotate(txt, (i + 1, tok1_cd9_percents.iloc[i]), xytext=(0, -10), textcoords='offset points', ha='center')
    """

    plt.title(f'Probabilities for CD: 0.5 - Idx {name}')
    plt.xlabel('Token')
    plt.ylabel('Probability (%)')
    plt.xticks(range(1, max_length + 1))
    plt.legend()
    plt.grid(True)
    #plt.show()
    plot_file = os.path.join(output_dir, f'plot_0.5_{name}.png')
    plt.savefig(plot_file)
    plot_files.append(plot_file)
    plt.close()


# Create an Excel file and insert the images into the sheet
excel_file = 'Filter_CD05_G.xlsx'
workbook = xlsxwriter.Workbook(excel_file)
worksheet = workbook.add_worksheet('Filter CD05 G')

# Insert images into the sheet
row = 0
for plot_file in plot_files:
    worksheet.insert_image(row, 0, plot_file)
    row += 20  # Adjust row as needed to prevent overlap

workbook.close()

print(f'Excel file {excel_file} created successfully with plots.')
