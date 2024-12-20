import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error loading data: {e}")

def check_shape(df):
    try:
        print("Shape of the DataFrame:", df.shape)
    except Exception as e:
        print(f"Error checking shape: {e}")

def get_summary(df):
    try:
        print("\nSummary of the DataFrame:")
        df.info()
    except Exception as e:
        print(f"Error getting summary: {e}")

def inspect_data(df):
    try:
        return df.head()
    except Exception as e:
        print(f"Error inspecting data: {e}")

def check_duplicates(df):
    try:
        duplicates = df.duplicated().sum()
        print(f'Number of duplicate rows: {duplicates}')
        return duplicates
    except Exception as e:
        print(f"Error checking duplicates: {e}")

def identify_unique_columns(df):
    try:
        unique_columns = [col for col in df.columns if df[col].nunique() == len(df)]
        print("Columns with unique values:")
        print(unique_columns)
        return unique_columns
    except Exception as e:
        print(f"Error identifying unique columns: {e}")

def check_missing_values(df):
    try:
        missing_values = df.isnull().sum()
        return missing_values
    except Exception as e:
        print(f"Error checking missing values: {e}")

def calculate_missing_percentage(df):
    try:
        missing_percentage = df.isnull().mean() * 100
        missing_percentage = missing_percentage[missing_percentage > 0]
        return missing_percentage
    except Exception as e:
        print(f"Error calculating missing percentage: {e}")

def remove_high_missing_cols(df, high_missing_cols):
    try:
        return df.drop(columns=high_missing_cols)
    except Exception as e:
        print(f"Error removing high missing columns: {e}")

def remove_unnecessary_cols(df, cols_to_remove):
    try:
        return df.drop(columns=cols_to_remove)
    except Exception as e:
        print(f"Error removing unnecessary columns: {e}")

def remove_rows_with_missing_values(df, cols_to_check):
    try:
        return df.dropna(subset=cols_to_check)
    except Exception as e:
        print(f"Error removing rows with missing values: {e}")

def placeholder_imputation(df, placeholder_impute_cols):
    try:
        df = df.assign(**{col: df[col].fillna(placeholder) for col, placeholder in placeholder_impute_cols.items()})
        return df
    except Exception as e:
        print(f"Error in placeholder imputation: {e}")

def mode_imputation(df, categorical_cols):
    try:
        df = df.assign(**{col: df[col].fillna(df[col].mode()[0]) for col in categorical_cols})
        return df
    except Exception as e:
        print(f"Error in mode imputation: {e}")

def verify_no_missing_values(df):
    try:
        missing_values_after = df.isnull().sum()
        print(missing_values_after[missing_values_after > 0])
    except Exception as e:
        print(f"Error verifying no missing values: {e}")

def visualize_outliers(df, numerical_cols):
    try:
        plt.figure(figsize=(15, 30))
        for i, col in enumerate(numerical_cols, 1):
            plt.subplot(len(numerical_cols), 1, i)
            sns.boxplot(x=df[col])
            plt.title(col)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Error visualizing outliers: {e}")

def identify_outliers(df, numerical_cols):
    try:
        outliers = {}
        for col in numerical_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers[col] = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        return outliers
    except Exception as e:
        print(f"Error identifying outliers: {e}")

def display_outliers(outliers):
    try:
        for col, outlier_data in outliers.items():
            print(f"{col}: {len(outlier_data)} outliers")
    except Exception as e:
        print(f"Error displaying outliers: {e}")

def save_cleaned_data(df, file_path):
    try:
        df.to_csv(file_path, index=False)
        print(f"Cleaned data saved to {file_path}")
    except Exception as e:
        print(f"Error saving cleaned data: {e}")