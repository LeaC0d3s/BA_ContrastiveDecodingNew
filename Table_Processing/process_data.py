import pandas as pd
import matplotlib.pyplot as plt
import os
import xlsxwriter

# Load the Excel file
file_path = 'prob_comp_table_full.xlsx'  # Replace with your actual file path
xls = pd.ExcelFile(file_path)

# Read the 'Blatt A - prob_comp_table_A_ful' table from row 2
table_a = pd.read_excel(xls, 'Blatt G - prob_comp_table_G_ful', header=2)

table_a.columns = [
        'Idx', 'tok1_int_b', 'tok1_b', 'tok1_b%', 'tok2_int_b', 'tok2_b', 'tok2_b%', 'tok3_int_b',
        'tok3_b', 'tok3_b%',
        'tok1_int_cd5', 'tok1_cd5', 'tok1_cd5%', 'tok2_int_cd5', 'tok2_cd5', 'tok2_cd5%', 'tok3_int_cd5',
        'tok3_cd5', 'tok3_cd5%',
        'tok1_int_cd9', 'tok1_cd9', 'tok1_cd9%', 'tok2_int_cd9', 'tok2_cd9', 'tok2_cd9%', 'tok3_int_cd9',
        'tok3_cd9', 'tok3_cd9%'
    ]

# Display the first few rows of table A to verify the structure
print("Initial table structure:\n", table_a.columns)

# Filter out rows where tok1_int_b = 13
filtered_table_a = table_a[table_a['tok1_int_b'] != 13]

# Group by 'Idx'
grouped = filtered_table_a.groupby('Idx')

filtered_table_a['tok1_b%'] = pd.to_numeric(filtered_table_a['tok1_b%'], errors='coerce')
filtered_table_a['tok1_cd5%'] = pd.to_numeric(filtered_table_a['tok1_cd5%'], errors='coerce')
filtered_table_a['tok1_cd9%'] = pd.to_numeric(filtered_table_a['tok1_cd9%'], errors='coerce')

colors = {
    'tok1_b%': 'blue',
    'tok1_cd5%': 'green',
    'tok1_cd9%': 'red'
}

# Create a directory to save the plots
output_dir = 'plots'
os.makedirs(output_dir, exist_ok=True)

plot_files = []
# For each group, create a diagram showing the probabilities
for name, group in grouped:
    #print(name, group)
    plt.figure(figsize=(12, 6))
    lengths = {
        'tok1_b%': len(group['tok1_b%'].dropna()),
        'tok1_cd5%': len(group['tok1_cd5%'].dropna()),
        'tok1_cd9%': len(group['tok1_cd9%'].dropna())
    }

    # Sort columns by length
    sorted_columns = sorted(lengths, key=lengths.get, reverse=True)

    max_length = 0

    for col in sorted_columns:
        print(col)
        if col in group.columns:
            percents = group[col].reset_index(drop=True)
            max_length = max(max_length, len(percents))
            plt.plot(range(1, len(percents) + 1), percents, label=col, marker='o', color=colors[col])
            if col == "tok1_b%":
                for i, txt in enumerate(group['tok1_b'].reset_index(drop=True)):
                    if percents.iloc[i] < 0.9:  # Probability below 80%
                        plt.annotate(txt, (i + 1, percents.iloc[i]), ha='right', fontsize=10, xytext=(0, -10), textcoords='offset points')
            if col == "tok1_cd5%":
                for i, txt in enumerate(group['tok1_cd5'].reset_index(drop=True)):
                    if percents.iloc[i] < 0.9:  # Probability below 80%
                        plt.annotate(txt, (i + 1, percents.iloc[i]), ha='right', fontsize=10, xytext=(0, -10), textcoords='offset points')
            if col == "tok1_cd9%":
                for i, txt in enumerate(group['tok1_cd9'].reset_index(drop=True)):
                    if percents.iloc[i] < 0.9:  # Probability below 80%
                        plt.annotate(txt, (i + 1, percents.iloc[i]), ha='right', fontsize=10, xytext=(0, -10), textcoords='offset points')
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

    plt.title(f'Probabilities for Idx {name}')
    plt.xlabel('Token')
    plt.ylabel('Probability (%)')
    plt.xticks(range(1, max_length + 1))
    plt.legend()
    plt.grid(True)
    #plt.show()
    plot_file = os.path.join(output_dir, f'plot_{name}.png')
    plt.savefig(plot_file)
    plot_files.append(plot_file)
    plt.close()


# Create an Excel file and insert the images into the sheet
excel_file = 'Filter_G.xlsx'
workbook = xlsxwriter.Workbook(excel_file)
worksheet = workbook.add_worksheet('Filter G')

# Insert images into the sheet
row = 0
for plot_file in plot_files:
    worksheet.insert_image(row, 0, plot_file)
    row += 20  # Adjust row as needed to prevent overlap

workbook.close()

print(f'Excel file {excel_file} created successfully with plots.')
