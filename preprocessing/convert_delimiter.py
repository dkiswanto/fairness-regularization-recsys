DATASET_DIR = 'ratings_min_30.dat'
OUTPUT_FILE = 'output.dat'
OUTPUT_DELIMITER = '\t'

file_data = open(DATASET_DIR)
out_file = open(OUTPUT_FILE, 'w')

for d in file_data:
    row = d.split('::')
    out_file.write(OUTPUT_DELIMITER.join(map(str, row)))

print("finished")
file_data.close()
out_file.close()
