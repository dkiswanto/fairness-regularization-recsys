import pandas as pd
import gzip


def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield eval(l)


def getDF(path):
    i = 0
    df = {}
    for d in parse(path):
        # remove unnecessary attribute
        if d.get('helpful'): del d['helpful']
        if d.get('reviewText'): del d['reviewText']
        if d.get('reviewerName'): del d['reviewerName']
        if d.get('summary'): del d['summary']
        if d.get('reviewTime'): del d['reviewTime']
        df[i] = d
        i += 1
        if i % 1000 == 0:
            print(i)
    return pd.DataFrame.from_dict(df, orient='index')


FILENAME = '/home/g40/reviews_Books_5.json.gz'
df = getDF(FILENAME)

# Save to CSV
OUTPUT_FILE = 'gzip_output.csv'
out_file = open(OUTPUT_FILE, 'w')
print('writing to csv....')
for data in df.itertuples():
    row = [data.reviewerID, data.asin, int(data.overall), data.unixReviewTime]
    # out_file.write("::".join(map(str, row)) + '\n') # using ::
    out_file.write(",".join(map(str, row)) + '\n') # using tab
    # print(row)

out_file.close()
