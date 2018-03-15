from datetime import datetime

import pandas as pd

from config import DATASET_DIR, DATASET_DELIMITER

filename = DATASET_DIR
MIN_RATING = 80

columns = ['user_id', 'item_id', 'rating', 'timestamp']
ratings = pd.read_table(filename, sep=DATASET_DELIMITER, header=None, names=columns, engine='python')

item_series = ratings.item_id.value_counts()
user_series = ratings.user_id.value_counts()

set_item_removed = set()
for idx, count in item_series.iteritems():
    if count < MIN_RATING:
        print("item_id {} remove with rating count {}".format(idx, count))
        set_item_removed.add(idx)
        # ratings = ratings[ratings.item_id != idx]
print('item remove by in set')
ratings = ratings[~ratings.item_id.isin(set_item_removed)]

set_user_removed = set()
for idx, count in user_series.iteritems():
    if count < MIN_RATING:
        print("user_id {} remove with rating count {}".format(idx, count))
        set_user_removed.add(idx)
        # ratings = ratings[ratings.user_id != idx]
print('user remove by in set')
ratings = ratings[~ratings.user_id.isin(set_user_removed)]


# print(ratings.user_id.value_counts())
# print(ratings.item_id.value_counts())
# print("total user : {}".format(ratings.user_id.nunique()))
# print("total item :{}".format(ratings.item_id.nunique()))
# print("total rating {}".format(ratings.count()))

USER_COLUMN, ITEM_COLUMN, RATING_COLUMN = 1, 2, 3

OUTPUT_FILE = "{}-{}.min".format("output", datetime.now())
out_file = open(OUTPUT_FILE, 'w')

for data in ratings.itertuples():
    row = [data[USER_COLUMN], data[ITEM_COLUMN], data[RATING_COLUMN], data[4]]
    # out_file.write("::".join(map(str, row)) + '\n') # using ::
    out_file.write(",".join(map(str, row)) + '\n') # using tab
    print(row)

out_file.close()