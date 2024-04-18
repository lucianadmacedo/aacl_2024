import pandas as pd
import numpy as np
from scipy.stats import ttest_ind

# Step 1: Read the spreadsheet into a pandas DataFrame
df = pd.read_csv("corhum_grammatical.csv")

# Step 2: Extract relevant information from the file names
df[['Field', 'journal', 'Section']] = df['filename'].str.split('.', expand=True).iloc[:, :3]

# Step 3: Define a function to calculate pooled standard deviation
def pooled_std(group1, group2):
    n1 = len(group1)
    n2 = len(group2)
    std1 = group1.std()
    std2 = group2.std()
    pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
    return pooled_std

# Step 4: Group the DataFrame by section
grouped_by_section = df.groupby('Section')

# Step 5: Calculate t-tests for each section against all other sections
for section_target, data_target in grouped_by_section:
    # Define target and reference groups
    group1 = data_target.select_dtypes(include=[np.number])
    group2 = df[~df['Section'].isin([section_target])].select_dtypes(include=[np.number])

    # Calculate means for target and reference groups
    means_target = group1.mean()
    means_ref = group2.mean()

    # Calculate pooled standard deviation
    pooled_std_dev = pooled_std(group1, group2)

    # Calculate MeanDiff
    mean_diff = means_target - means_ref

    # Calculate Cohen's d
    cohen_d = mean_diff / pooled_std_dev

    # Take the absolute value of Cohen's d and round to 4 decimal places
    abs_cohen_d = np.round(np.abs(cohen_d), 4)

    # Calculate effect size
    effect_size = pd.cut(abs_cohen_d, bins=[-np.inf, 0, 0.2, 0.5, 0.8, np.inf],
                         labels=['none', 'negligible', 'small', 'medium', 'large'], right=False)

    # Round MeanDiff, Cohen's d and pooled standard deviation to 4 decimal places
    mean_diff = np.round(mean_diff, 4)
    pooled_std_dev = np.round(pooled_std_dev, 4)
    cohen_d = np.round(cohen_d, 4)

    # Create a DataFrame containing the means, pooled standard deviation, MeanDiff, Cohen's d, and effect size
    result_df = pd.DataFrame({'Section_Target': [section_target] * len(means_target),
                              'Section_Ref': ['All other sections'] * len(means_target),
                              'Column': means_target.index,
                              'Mean_Target': np.round(means_target.values, 4),
                              'Mean_Ref': np.round(means_ref.values, 4),
                              'Pooled_Std': pooled_std_dev.values,
                              'MeanDiff': mean_diff.values,
                              'CohenD': cohen_d.values,
                              'Effect_Size': effect_size})

    # Sort the DataFrame by 'CohenD' column in descending order
    result_df.sort_values(by='CohenD', ascending=False, inplace=True)

    # Save the result to a new CSV file
    result_df.to_csv(f"{section_target}_vs_all_other_sections.csv", index=False)
