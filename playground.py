import numpy as np

from recommender.fairness_reg_als import FairnessRegALS

R = np.array([
    [4, 3, 3, 4, 5, 4, 3, 3, 4, 5, 4, 3, 3, 4, 5, 4, 3, 3, 4, 0],
    [4, 0, 0, 1, 0, 0, 0, 1, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 0, 5, 0, 5, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 4, 1, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 5, 4, 0, 0, 5, 1, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [5, 3, 4, 3, 3, 0, 4, 1, 5, 0, 2, 0, 5, 0, 5, 5, 0, 4, 5, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 5],
    [5, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 4, 3, 3, 0, 4, 1, 5, 0, 2, 5, 0, 5, 0, 5, 4, 0, 0, 5]
])

# als = FairnessRegALS(matrix_R=R, n_factor=6)
# als.train_data(iteration=10)
# als.save_data('save_data/playground')
#
# recommendation_list = als.top_n_recommendation(user_idx=1, n=5)
# print(recommendation_list)

watched = np.array([5, 0, 0, 3, 0, 0, 4, 1, 5, 0, 2, 0, 5, 0, 5, 5, 0, 4, 5, 0])
non_watched = np.where(watched == 0)

R_user = np.array([5, 3, 4, 3, 3, 0, 4, 1, 5, 0, 2, 0, 5, 0, 5, 5, 1, 4, 5, 0])
print(R_user[non_watched])

