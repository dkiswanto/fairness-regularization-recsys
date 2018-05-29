from config import DATASET_DIR, MODEL_LOCATION
from recommender.evaluation import medium_tail_test, apt_test, ndcg_test
from recommender.fairness_reg_als import FairnessRegALS
from recommender.util import load_dataset, divide_item_popularity, dataframe_to_matrix

"""
    IMPORTANT MESSAGE
    ALWAYS CREATE MODEL IN training.py
"""


def main(session):

    # load dataset
    ratings_df = load_dataset(DATASET_DIR)
    R_ratings = dataframe_to_matrix(ratings_df)

    # divide set popularity
    short_head, medium_tail = divide_item_popularity(ratings_df)

    # load recommender model
    # WARNING: ONLY CREATE MODEL IN training.py
    model_location = MODEL_LOCATION + "-{}".format(session)
    als = FairnessRegALS.load_data(model_location)

    if als is None:
        raise Exception('recommender model {} not found, please check directory'.format(model_location))

    print("model {} loaded".format(model_location))

    # prepare user test
    user_idx_test = als.df_test.user_id.unique()

    # evaluation medium count
    medium_count = medium_tail_test(als, user_idx_test, medium_tail)
    print("medium tail count {}".format(medium_count))

    # evaluation apt
    apt_value = apt_test(als, user_idx_test, medium_tail)
    print("apt percentage {}".format(apt_value))

    # evaluation ndcg
    ndcg_value = ndcg_test(als, user_idx_test, R_ratings)
    print("ndcg value {}".format(ndcg_value))

    # model info
    print("lambda reg {}".format(als.lambda_reg))
    print("n factor {}".format(als.n_factor))

if __name__ == '__main__':
    for session in range(1, 10):
        main(session)
        print("session testing: {} done\n".format(session))