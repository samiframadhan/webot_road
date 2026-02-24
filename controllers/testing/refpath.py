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

def remove_duplicates(input, output, column):
    column_index = 1  # The index of the column to check for duplicates (0 is the first column)

    with open(input, 'r', newline='', encoding='utf-8') as in_file, \
        open(output, 'w', newline='', encoding='utf-8') as out_file:
        
        reader = csv.reader(in_file)
        writer = csv.writer(out_file)
        
        # Optional: Handle the header row separately so it doesn't get treated as data
        header = next(reader)
        writer.writerow(header)
        
        seen = set()
        rows_removed = 0

        for row in reader:
            # Extract the specific value we want to check
            identifier = row[column_index]
            
            if identifier not in seen:
                writer.writerow(row)
                seen.add(identifier)
            else:
                rows_removed += 1

    print(f"Process complete. Removed {rows_removed} rows based on column index {column_index}.")

# Usage
keep_columns('test.csv', 'filtered_data.csv', ['timestamp', 'smooth_x', 'smooth_y'])
# remove_duplicates('filtered_data.csv', "final.csv")