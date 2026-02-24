import csv

input_file = 'ref_path2.csv'
output_file = 'ref_path3.csv'

column_index = 1  # The index of the column to check for duplicates (0 is the first column)

with open(input_file, 'r', newline='', encoding='utf-8') as in_file, \
     open(output_file, 'w', newline='', encoding='utf-8') as out_file:
    
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