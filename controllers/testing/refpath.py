import csv
import os

def keep_columns(input_file, output_file, columns_to_keep):
    with open(input_file, mode='r', newline='', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        
        # Define the headers for the new file
        fieldnames = columns_to_keep
        
        with open(output_file, mode='w', newline='', encoding='utf-8') as outfile:
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            
            # Write the header row first
            writer.writeheader()
            
            # Iterate through the rows, only writing the allowed keys
            for row in reader:
                # Filter the dictionary to only include specified columns
                filtered_row = {k: row[k] for k in columns_to_keep if k in row}
                writer.writerow(filtered_row)

# Usage
keep_columns('test.csv', 'filtered_data.csv', ['timestamp', 'smooth_x', 'smooth_y'])