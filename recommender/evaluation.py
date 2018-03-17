import math
from operator import itemgetter

import numpy as np

from recommender.fairness_reg_als import FairnessRegALS


# def disc(i):
#     return 1 if i <= 2 else math.log2(i)
#
#
# def ndcg(R_dataset, recommendation_list, user_id):
#     # That is, the discount is set to be zero for ranks larger than k.
#     # Such NDCG measure is usually referred to as NDCG@k.
#     dcg = 0.0
#
#     for i, item in enumerate(recommendation_list):
#         actual_value = R_dataset[user_id - 1, item - 1]
#         dcg += actual_value / disc(i + 1)
#
#     # compute perfect dcg
#     all_r_user = R_dataset[user_id - 1]
#     rating_value = all_r_user[recommendation_list - 1]
#     indices = list(zip(recommendation_list, rating_value))
#     perfect_list = sorted(indices, key=itemgetter(1), reverse=True)
#
#     dcg_perfect = 0.0
#     for i, item in enumerate(perfect_list):
#         idx, rating = item
#         dcg_perfect += rating / disc(i + 1)
#
#     # Discount Account
#     # 1 / log(1+r), log is base 2
#     if dcg_perfect == 0:
#         return 1
#     return dcg / dcg_perfect


def dcg_at_k(r, k, method=0):
    rating = np.asfarray(r)[:k]
    if rating.size:
        if method == 0:
            return rating[0] + np.sum(rating[1:] / np.log2(np.arange(2, rating.size + 1)))
        elif method == 1:
            return np.sum(rating / np.log2(np.arange(2, rating.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def ndcg_at_k(r, k, method=0):
    """
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
    Returns:
        Normalized discounted cumulative gain
    """
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max


def ndcg_test(recommender: FairnessRegALS, user_test, R_dataset):
    num_user_test = len(user_test)
    sum_ndcg = 0
    for user_id in user_test:

        recommendation_list = recommender.top_n_recommendation(user_id, 10, return_index=True)

        # all item that user has been rating
        user_idx = recommender.user_index.get_loc(user_id)
        all_r_user = R_dataset[user_idx]

        rating_value = all_r_user[recommendation_list]

        sum_ndcg += ndcg_at_k(rating_value, 10, 1)

    return sum_ndcg / num_user_test


def medium_tail_test(recommender: FairnessRegALS, user_test, medium_tail_set):
    """
    we also measured medium-tail coverage, 
    the total number of items from the medium-tail set 
    that are recommended to any user in the test data. 
    'Controlling Popularity Bias in Learning-to-Rank Recommendation'- RecSys 17
    """
    medium_data = {}
    for user in user_test:
        recommendation_list = recommender.top_n_recommendation(user, 10)
        # print("user {} recommend {}".format(user, recommendation_list))
        for item in recommendation_list:
            if item in medium_tail_set:
                if medium_data.get(item) is None:
                    medium_data[item] = 1
                else:
                    medium_data[item] += 1

    medium_count = 0
    total_user_test = len(user_test)
    for key in medium_data:
        # it means, every medium tail item has appear in every user test
        if medium_data.get(key) == total_user_test:
            medium_count += 1
    print("max medium tail item", max(medium_data.values()))
    print("total user test", total_user_test)
    return medium_count


def apt_test(recommender: FairnessRegALS, user_test, medium_tail_set):

    num_user_test = len(user_test)
    sum_medium = 0
    for user in user_test:

        recommendation_list = recommender.top_n_recommendation(user, 10)

        # medium_item = set.intersection(set(recommendation_list), medium_tail_set)
        # print(user, len(medium_item), len(recommendation_list))
        total_medium_item = 0
        for item in recommendation_list:
            if item in medium_tail_set:
                total_medium_item += 1

        sum_medium += total_medium_item / len(recommendation_list)
    return sum_medium / num_user_test
