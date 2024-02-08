import pandas as pd

# Step 1: Read the spreadsheet into a pandas DataFrame
df = pd.read_excel("/Users/lucianadiasdemacedo/Documents/cnpq_inpact_humanities/teste_planilha_input.xlsx")

# Step 2: Extract relevant information from the file names
df[['Field', 'journal', 'Section']] = df['filename'].str.split('.', expand=True).iloc[:, :3]

# Step 3: Calculate the total number of sections within each field
total_sections_per_field = df.groupby('Field')['Section'].nunique().reset_index(name='NumSectionsTotal')

# Step 4: Group the DataFrame by field of study and section
grouped_df = df.groupby(['Field', 'Section'])

# Step 5: Calculate the desired statistics for each group
result_df = grouped_df['pasttnse'].agg(['sum', 'mean', 'std'])

# Step 6: Create a new DataFrame to store the results
result_df.reset_index(inplace=True)

# Print the resulting DataFrame
print(result_df)

# Step 7: Save the results to a new Excel file
result_df.to_excel("/Users/lucianadiasdemacedo/Documents/cnpq_inpact_humanities/results_teste.xlsx", index=False)
