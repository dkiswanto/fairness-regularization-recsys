import numpy as np

from recommender.fairness_reg_als import FairnessRegALS


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
    medium_data = set()
    for user in user_test:
        recommendation_list = recommender.top_n_recommendation(user, 10)
        # print("user {} recommend {}".format(user, recommendation_list))
        for item in recommendation_list:
            if item in medium_tail_set and is_test_data(recommender.df_test, user, item):
                medium_data.add(item)

    return len(medium_data)


def apt_test(recommender: FairnessRegALS, user_test, medium_tail_set):

    num_user_test = len(user_test)
    sum_medium = 0
    for user in user_test:

        recommendation_list = recommender.top_n_recommendation(user, 10)

        total_medium_item = 0
        rec_count = 0
        for item in recommendation_list:
            if is_test_data(recommender.df_test, user, item):
                rec_count += 1
                if item in medium_tail_set:
                    total_medium_item += 1

        if rec_count != 0:
            sum_medium += total_medium_item / rec_count
        else:
            num_user_test -= 1

    return sum_medium / num_user_test


def is_test_data(df, user_id, item_id):
    result = df.loc[(df.user_id == user_id) & (df.item_id == item_id)]
    return len(result) == 1
