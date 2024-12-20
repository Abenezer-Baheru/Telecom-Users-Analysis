import pandas as pd
from preprocessing_module import (
    load_data, check_shape, get_summary, inspect_data, check_duplicates,
    identify_unique_columns, check_missing_values, calculate_missing_percentage,
    remove_high_missing_cols, remove_unnecessary_cols, remove_rows_with_missing_values,
    placeholder_imputation, mode_imputation, verify_no_missing_values,
    visualize_outliers, identify_outliers, display_outliers, save_cleaned_data
)

def main():
    try:
        # Load the data
        data_path = '../src/data/telecom_users_data_source.csv'
        df = load_data(data_path)

        # Check the shape of the data
        check_shape(df)

        # Get a summary of the data
        get_summary(df)

        # Inspect the data
        inspect_data(df)

        # Check for duplicates
        check_duplicates(df)

        # Identify columns with unique values
        identify_unique_columns(df)

        # Check for missing values
        check_missing_values(df)

        # Calculate the percentage of missing values for each column
        missing_percentage = calculate_missing_percentage(df)
        print(missing_percentage)

        # Remove columns with high missing values (more than 50%)
        high_missing_cols = [
            'HTTP DL (Bytes)', 'HTTP UL (Bytes)', 'Nb of sec with 125000B < Vol DL', 
            'Nb of sec with 1250B < Vol UL < 6250B', 'Nb of sec with 31250B < Vol DL < 125000B', 
            'Nb of sec with 37500B < Vol UL', 'Nb of sec with 6250B < Vol DL < 31250B', 
            'Nb of sec with 6250B < Vol UL < 37500B'
        ]
        df = remove_high_missing_cols(df, high_missing_cols)

        # Remove unnecessary columns for Task 1 and Task 2
        unnecessary_cols = ['Avg RTT DL (ms)', 'Avg RTT UL (ms)', 'TCP DL Retrans. Vol (Bytes)', 'TCP UL Retrans. Vol (Bytes)']
        df = remove_unnecessary_cols(df, unnecessary_cols)

        # Remove rows with missing values for specified columns
        remove_missing_cols = [
            'Dur. (ms)', 'DL TP < 50 Kbps (%)', '50 Kbps < DL TP < 250 Kbps (%)', 
            '250 Kbps < DL TP < 1 Mbps (%)', 'DL TP > 1 Mbps (%)', 'UL TP < 10 Kbps (%)', 
            '10 Kbps < UL TP < 50 Kbps (%)', '50 Kbps < UL TP < 300 Kbps (%)', 'UL TP > 300 Kbps (%)',
            'Start', 'Start ms', 'End', 'End ms', 'Nb of sec with Vol DL < 6250B', 'Nb of sec with Vol UL < 1250B'
        ]
        df = remove_rows_with_missing_values(df, remove_missing_cols)

        # Placeholder Imputation for Last Location Name and Unique Identifiers
        placeholder_impute_cols = {
            'Last Location Name': 'missing_last_location',
            'Bearer Id': 'missing_bearer_Id',
            'IMSI': 'missing_IMSI',
            'MSISDN/Number': 'missing_MSISDN/Number',
            'IMEI': 'missing_IMEI'
        }
        df = placeholder_imputation(df, placeholder_impute_cols)

        # Mode Imputation for Categorical Columns
        categorical_cols = ['Handset Manufacturer', 'Handset Type']
        df = mode_imputation(df, categorical_cols)

        # Verify that there are no more missing values
        verify_no_missing_values(df)

        # Check the shape of the data after handling missing values
        check_shape(df)

        # List of numerical columns to check for outliers
        numerical_cols = [
            'Dur. (ms)', 'DL TP < 50 Kbps (%)', '50 Kbps < DL TP < 250 Kbps (%)', 
            '250 Kbps < DL TP < 1 Mbps (%)', 'DL TP > 1 Mbps (%)', 'UL TP < 10 Kbps (%)', 
            '10 Kbps < UL TP < 50 Kbps (%)', '50 Kbps < UL TP < 300 Kbps (%)', 'UL TP > 300 Kbps (%)'
        ]

        # Visualize the data using box plots
        visualize_outliers(df, numerical_cols)

        # Identify outliers using the IQR method
        outliers = identify_outliers(df, numerical_cols)

        # Display the number of outliers for each column
        display_outliers(outliers)

        # Save the cleaned data
        cleaned_data_path = '../src/data/cleaned_telecom_users_data.csv'
        save_cleaned_data(df, cleaned_data_path)

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
