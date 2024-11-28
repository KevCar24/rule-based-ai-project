import pandas as pd

diabetes_df = pd.read_csv('diabetes_prediction_dataset.csv')

# convert categorical variables to numeric (used ChatGPT for these 2 lines)
diabetes_df['gender'] = pd.Categorical(diabetes_df['gender']).codes
diabetes_df['smoking_history'] = pd.Categorical(diabetes_df['smoking_history']).codes

# calculate correlations with pandas library
correlation_matrix = diabetes_df.corr()

# sort the correlations so they are easier to read
sorted_correlations = correlation_matrix['diabetes'].sort_values(ascending=False)

print("Correlations with Diabetes:")
print(sorted_correlations)
