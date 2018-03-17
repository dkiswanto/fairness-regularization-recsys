import pandas as pd

from config import DATASET_DELIMITER


def load_dataset(data_dir):
    columns = ['user_id', 'item_id', 'rating', 'timestamp']
    df = pd.read_table(data_dir, sep=DATASET_DELIMITER, header=None, names=columns,
                       engine='python')
    print('load data set done')
    return df


def divide_item_popularity(dataset_frame):
    # set short-head rating limit bound
    SHORT_HEAD_BOUND = 85

    item_series = dataset_frame.item_id.value_counts()
    short_head = set()
    medium_tail = set()
    for item_id, count in item_series.iteritems():
        if count > SHORT_HEAD_BOUND:
            short_head.add(item_id)
        else:
            medium_tail.add(item_id)
    print('divide item popularity done with sort-head {} & medium-tail {}'
          .format(len(short_head), len(medium_tail)))
    return short_head, medium_tail


def dataframe_to_matrix(df, with_index=False):
    matrix = df.pivot_table(columns=['item_id'], index=['user_id'], values='rating')
    if with_index:
        return matrix.fillna(0).as_matrix(), matrix.index, matrix.columns
    else:
        return matrix.fillna(0).as_matrix()
