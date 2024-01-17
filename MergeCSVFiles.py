import pandas as pd
import glob

# Specify the path to the directory containing your CSV files (replace 'your_directory' with the actual path)
directory_path = 'C:\\Users\\Thesis\\Downloads\\DDoS-HTTP_Flood\\DDoS-HTTP_Flood\\DDoS-HTTP_Flood\\'

# Get a list of all CSV files in the directory
csv_files = glob.glob(directory_path + '\\*.csv')


# Initialize an empty DataFrame to store the merged data
merged_df = pd.DataFrame()

# Loop through each CSV file and append its data to the merged DataFrame
for file in csv_files:
    original_df = pd.read_csv(file)
    print("Reading file: ", file)
    # Get % of the dataset using the sample method
    sampled_df = original_df.sample(frac=0.0035, random_state=42)  # Set random_state for reproducibility

    # Save the sampled DataF
    merged_df = pd.concat([merged_df, sampled_df], ignore_index=True)
    print("Merged file: ", file)
# Sort the DataFrame based on an integer column (replace 'your_integer_column' with the actual column name)
print("Sorting dataframe")
merged_df.sort_values(by='ts', inplace=True)
print("Sorting dataframe done")
# Save the sorted and merged DataFrame to a new CSV file (replace 'merged_data_sorted.csv' with your desired file name)
print("Saving merged dataframe")
merged_df.to_csv(directory_path + '\\DDoS-HTTP_Flood_035percent.csv__', index=False)
print("Saving merged dataframe done")
# Display information about the sorted and merged DataFrame
print("Sorted and Merged DataFrame:")
print(merged_df.head())
