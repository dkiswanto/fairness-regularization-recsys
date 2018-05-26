from datetime import datetime

import numpy as np
import pandas as pd

from recommender.util import dataframe_to_matrix, divide_item_popularity


class FairnessRegALS:
    def __init__(self, df_train, n_factor, user_factor=None, item_factor=None):

        # get data from dataframe
        self.data_frame = df_train
        R_matrix, user_index, item_index = dataframe_to_matrix(df_train, with_index=True)
        self.R = R_matrix
        self.user_index = user_index
        self.item_index = item_index

        self.R_prediction = None
        self.n_user, self.n_item = self.R.shape

        # common parameter
        self.n_factor = n_factor

        # rank als parameter
        self.is_support_weight = False

        # initialize user (P) & item (Q) latent model
        if user_factor is None and item_factor is None:
            self.P = np.random.rand(self.n_user, self.n_factor)
            self.Q = np.random.rand(self.n_item, self.n_factor)
        else:
            self.P = user_factor
            self.Q = item_factor

        # DEBUGGING PURPOSE ONLY, CONSTANT FACTOR CAN RAISING ERROR!.
        # self.constant_latent_factor()

        # check matrix shape dimension
        if self.P.shape != (self.n_user, self.n_factor) \
                or self.Q.shape != (self.n_item, self.n_factor):
            raise ValueError

        # building support weight
        self.support_weight_vector = np.zeros(self.n_item)
        self.support_weight_sum = 0
        for item_idx in range(self.n_item):
            support_value = np.count_nonzero(self.R[:, item_idx]) if self.is_support_weight else 1
            self.support_weight_vector[item_idx] = support_value
            self.support_weight_sum += support_value

        # fairness regularization attribute
        self.pre = False
        self.lambda_reg = 0.004
        self.div_dot = np.zeros(self.n_item)

        # divide popularity-rank
        self.short_head, self.medium_tail = divide_item_popularity(df_train)

    def train_data(self, iteration, directory=None):
        for iterate in range(iteration):
            print("als train_data iteration {} at {}".format(iterate + 1, datetime.now()))

            ###################################################
            # P STEP: UPDATE USER FACTOR
            ###################################################

            # q̃ = QTs,
            # Ã = QT diag(s)Q
            q_tilde = np.zeros(self.n_factor)
            A_tilde = np.zeros((self.n_factor, self.n_factor))
            for item in range(self.n_item):
                qj = self.Q[item, :]
                sj = self.support_weight_vector[item]
                q_tilde += qj * sj
                A_tilde = A_tilde + np.outer(qj, qj)

            # for u ← 1,..., U do ;
            # TODO: RE-CHECK THIS, IT HAS SIDE-EFFECT IN DATA-SET (R)
            cus = [i for i in range(self.n_user) if np.any(self.R[i, :])]
            for user in cus:
                A_bar = np.zeros((self.n_factor, self.n_factor))
                q_bar = np.zeros(self.n_factor)
                b_bar = np.zeros(self.n_factor)
                b_tilde = np.zeros(self.n_factor)

                Ru = self.filter_row(self.R[user])  # OK
                I_bar = len(Ru)  # OK
                r_tilde, r_bar = 0.0, 0.0  # OK

                for i, rui in Ru:
                    qi = self.Q[i]  # OK

                    A_bar = A_bar + np.outer(qi, qi)

                    q_bar = q_bar + qi
                    b_bar = b_bar + (qi * rui)
                    si = self.support_weight_vector[i]
                    r_tilde += si * rui
                    r_bar += rui
                    b_tilde = b_tilde + qi * (si * rui)

                M = A_bar * self.support_weight_sum \
                    - (np.outer(q_bar, q_tilde)) \
                    - (np.outer(q_tilde, q_bar)) \
                    + (A_tilde * I_bar)

                y = b_bar * self.support_weight_sum \
                    - (q_bar * r_tilde) \
                    - (q_tilde * r_bar) \
                    + (b_tilde * I_bar)

                pu = np.linalg.inv(M).dot(y)
                self.P[user, :] = pu

            ###################################################
            # Q STEP: UPDATE ITEM FACTOR
            ###################################################

            # FAIRNESS REGULARIZATION 1
            if not self.pre:
                for id_i in self.item_index:
                    idx = self.item_index.get_loc(id_i)
                    distance_items = 0
                    for id_j in self.item_index:
                        distance_items += self.ilbu_distance(id_i, id_j)
                    self.div_dot[idx] = distance_items
                self.pre = True
            # END REGULARIZATION 1

            # k, v -> int, double
            map_p1_tensor = {}
            map_p3_tensor = {}
            map_b_tensor = {}

            # k, v -> int, []
            map_p2_tensor = {}

            # for each user
            for user in cus:
                Ru = self.filter_row(self.R[user])  # this is okay

                sum_p1_tensor = 0.0
                sum_p3_tensor = 0.0
                sum_b_tensor = len(Ru)
                sum_p2_tensor = np.zeros(self.n_factor)

                for j, ruj in Ru:
                    sj = self.support_weight_vector[j]
                    sum_p1_tensor += sj * ruj
                    sum_p3_tensor += ruj

                    sum_p2_tensor = sum_p2_tensor + self.Q[j]

                map_p1_tensor[user] = sum_p1_tensor
                map_p3_tensor[user] = sum_p3_tensor
                map_b_tensor[user] = sum_b_tensor
                map_p2_tensor[user] = sum_p2_tensor

            # REGULARIZATION 2
            # dist = lambda param1, param2: param1 + param2
            sum_d_q = np.zeros((self.n_item, self.n_factor))
            for id_i in self.item_index:
                idx_i = self.item_index.get_loc(id_i)
                for id_j in self.item_index:
                    idx_j = self.item_index.get_loc(id_j)

                    # update sum_d_q
                    sum_d_q[idx_i] = sum_d_q[idx_i] + self.ilbu_distance(id_i, id_j) * self.Q[idx_j]

                    # method 2 update sum_d_q, unchecked & untested
                    # for x, y in np.nditer([sum_d_q[item_i], self.Q[item_j]], op_flags=['readwrite']):
                    #    x[...] = dist(x, y)

            # END REGULARIZATION 2

            # for each item
            for item in range(self.n_item):
                A_bar = np.zeros((self.n_factor, self.n_factor))
                A_tensor = np.zeros((self.n_factor, self.n_factor))
                b_bar = np.zeros(self.n_factor)

                p1_tensor = np.zeros(self.n_factor)
                p3_tensor = np.zeros(self.n_factor)
                b_tensor = np.zeros(self.n_factor)
                p2_tensor = np.zeros(self.n_factor)

                si = self.support_weight_vector[item]
                for user in cus:
                    pu = self.P[user]
                    rui = self.R[user, item]

                    pp = np.outer(pu, pu)  # 6x6 indeed
                    A_bar += pp

                    p2_tensor = p2_tensor + pp.dot(map_p2_tensor.get(user))
                    A_tensor = A_tensor + (pp * map_b_tensor.get(user))
                    p3_tensor = p3_tensor + pu * map_p3_tensor.get(user)

                    if rui > 0:
                        b_bar += pu * rui
                        p1_tensor += pu * map_p1_tensor.get(user)
                        b_tensor += pu * (rui * map_b_tensor.get(user))

                M = (A_bar * self.support_weight_sum) + (A_tensor * si)  # THIS IS DOPE
                y = A_bar.dot(q_tilde) \
                    + (b_bar * self.support_weight_sum) \
                    - p1_tensor \
                    + (p2_tensor * si) \
                    - (p3_tensor * si) \
                    + (b_tensor * si)

                # dope variable checked
                # b_bar, p1_tensor, p2_tensor, p3_tensor, b_tensor

                # REGULARIZATION 3
                # div_reg = np.zeros(self.n_factor)
                div_qi = self.Q[item]
                div_reg = sum_d_q[item]
                div_reg = div_reg - div_qi * self.div_dot[item]
                y = y + (self.lambda_reg * div_reg)
                # END REGULARIZATION 3

                qi = np.linalg.inv(M).dot(y)
                self.Q[item, :] = qi

            if directory is not None:
                print("model saved to {} at {}".format(directory, datetime.now()))
                self.save_data(directory)

        # build matrix prediction after training
        self.R_prediction = self.P.dot(self.Q.T)

    def predict(self, user_id, item_id):
        """
        :return: int number of rating predicted value
        """
        user_idx = self.user_index.get_loc(user_id)
        item_idx = self.item_index.get_loc(item_id)
        return self.P[user_idx, :].dot(self.Q[item_idx, :].T)

    def constant_latent_factor(self):
        """
        :update self.P & self.Q set to hardcoded constant data (not random)
        for validate & check algorithm purposes
        """
        P = [
            [0.389029, 0.37083, 0.138663, 0.330615, 0.32092, 0.023535],
            [0.096261, 0.331684, 0.180554, 0.082785, 0.163605, 0.224078],
            [0.279756, 0.343019, 7.446235, 0.365193, 0.331263, 0.120187],
            [0.260689, 0.344886, 0.209619, 0.181969, 0.329431, 0.242141],
            [0.171901, 0.108108, 0.239654, 0.168331, 0.328869, 0.102185],
            [0.38645, 0.302814, 0.031465, 0.096569, 0.048541, 0.364143],
            [0.05143, 0.382133, 0.202822, 0.334308, 0.107831, 0.005185],
            [0.07864, 0.200014, 0.206945, 0.181241, 0.37352, 0.191107],
            [0.083158, 0.172389, 0.039392, 0.193155, 0.233606, 0.029406],
            [0.110443, 0.344517, 0.212825, 0.331681, 0.219417, 0.009429]
        ]
        Q = [
            [0.104935, 0.116233, 0.240898, 0.168227, 0.228036, 0.154922],
            [0.070432, 0.082777, 0.115677, 0.020532, 0.015013, 0.034276],
            [0.250992, 0.113983, 0.175589, 0.016332, 0.327246, 0.315426],
            [0.366608, 0.331832, 0.266856, 0.289445, 0.293623, 0.400847],
            [0.350481, 0.05453, 0.246224, 0.012964, 0.106064, 0.164229],
            [0.152736, 0.091174, 0.245523, 0.168374, 0.099727, 0.229383],
            [0.359561, 0.092727, 0.380441, 0.271307, 0.154715, 0.040448],
            [0.131791, 0.355018, 0.260917, 0.338999, 0.241205, 0.204367],
            [0.060718, 0.353894, 0.283345, 0.327494, 0.102246, 0.211674],
            [0.386727, 0.284644, 0.347581, 0.090599, 0.380868, 0.073491],
            [0.339515, 0.333499, 0.146307, 0.34995, 0.270374, 0.161347],
            [0.242252, 0.35722, 0.259336, 0.162833, 0.215778, 0.094054],
            [0.11422, 0.119899, 0.034264, 0.103586, 0.406119, 0.132876],
            [0.068325, 0.356027, 0.269967, 0.235559, 0.05393, 0.046646],
            [0.006628, 0.321812, 0.333292, 0.082942, 0.353736, 0.071929],
            [0.024762, 0.20233, 0.32691, 0.324795, 0.078684, 0.194354],
            [0.374919, 0.335062, 0.14201, 0.403652, 0.095524, 0.201799],
            [0.264057, 0.367631, 0.343755, 0.159355, 0.079492, 0.174732],
            [0.2559, 0.027917, 0.237678, 0.176706, 0.118951, 0.064477],
            [0.029317, 0.037504, 0.30339, 0.165502, 0.381737, 0.197163]
        ]
        self.P = np.array(P)
        self.Q = np.array(Q)

    def matrix_prediction(self):
        if self.R_prediction is None:
            self.R_prediction = self.P.dot(self.Q.T)
        return self.R_prediction

    def top_n_recommendation(self, user_id, n,
                             return_index=False,
                             with_index=False,
                             with_reviewed=True):
        """
        :return: slice of n item_idx
        """
        user_idx = self.user_index.get_loc(user_id)

        if self.R_prediction is None:
            self.matrix_prediction()

        if not with_reviewed:
            non_watched_item = self.R_prediction[user_idx]
        else:
            watched = self.R[user_idx]
            non_watched_index = np.where(watched == 0)
            non_watched_item = self.R_prediction[user_idx][non_watched_index]

        index_sorted = np.argsort(non_watched_item)[-n:]

        # reversed because argsort cannot desc
        rec_item_idx = np.array(list(reversed(index_sorted)))

        # TODO: return_index option not awesome, will deprecated this
        if return_index:
            return rec_item_idx
        elif with_index:
            return [self.item_index[item] for item in rec_item_idx], rec_item_idx
        else:
            # convert list recommendation to id
            return [self.item_index[item] for item in rec_item_idx]

    def save_data(self, directory):
        np.save(directory + "/P.npy", self.P)
        np.save(directory + "/Q.npy", self.Q)
        self.data_frame.to_pickle(directory + '/data_frame.pkl')
        np.save(directory + "/n_factor.npy", self.n_factor)

    def ilbu_distance(self, i, j):
        """
        :param i: id item 1
        :param j: id item 2
        :return: for any pair of items i and j, dist(i,j)=1 if i & j are in the same set, 0 otherwise
        """
        if (i in self.short_head) and (j in self.short_head):
            return 1
        elif (i in self.medium_tail) and (j in self.medium_tail):
            return 1
        else:
            return 0

    # noinspection PyBroadException
    @staticmethod
    def load_data(directory):
        try:
            P = np.load(directory + "/P.npy")
            Q = np.load(directory + "/Q.npy")
            # R = np.load(directory + "/R.npy")
            data_frame = pd.read_pickle(directory + '/data_frame.pkl')
            n_factor = np.load(directory + "/n_factor.npy")

            return FairnessRegALS(data_frame, n_factor, P, Q)
        except Exception:
            return None

    @staticmethod
    def filter_row(vector):
        return [(i, j) for i, j in enumerate(vector) if j != 0]
