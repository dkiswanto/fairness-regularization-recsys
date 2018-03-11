import numpy as np

class MatrixFactorization:
    def __init__(self, R, latent_factor, iterations, alpha=0.1):

        """
        :param R: numpy.ndarray, matrix rating
        :param latent_factor:  int, dimension of latent factor model User & Item
        :param iterations: int, number of factorize (learning) iteration
        :param alpha: float, learning rate
        """

        self.R = R
        self.latent_factor_k = latent_factor
        self.iterations = iterations
        self.alpha = alpha  # learning-rate

        # unpack data total user & item
        self.n_user, self.n_item = R.shape

        # initialize random latent model
        self.P = np.random.rand(self.n_user, self.latent_factor_k)
        self.Q = np.random.rand(self.n_item, self.latent_factor_k)

    def factorize(self):

        # generate training data from matrix rating (R)
        training_data = []
        for u in range(self.n_user):
            for i in range(self.n_item):
                if self.R[u, i] > 0:
                    training_data.append((u, i, self.R[u, i]))

        # run matrix factorization
        for i in range(self.iterations):

            # run gradient descent (stochastic)
            self.sgd(training_data)

            # count error measure (error measure)
            print("iteration {} - with error (mse) {}".format(i+1, self.mse()))

    def sgd(self, training_data):
        for u, i, r in training_data:
            prediction = self.rating_prediction(u, i)
            e = r - prediction

            self.P[u, :] += self.alpha * (e * self.Q[i, :])
            self.Q[i, :] += self.alpha * (e * self.P[u, :])

    def rating_prediction(self, user_id, item_id):
        return self.P[user_id, :].dot(self.Q[item_id, :].T)

    def matrix_prediction(self):
        return self.P.dot(self.Q.T)

    def mse(self):
        xs, ys = self.R.nonzero()
        predicted_matrix = self.matrix_prediction()
        error = 0
        for x, y in zip(xs, ys):
            error += (self.R[x, y] - predicted_matrix[x, y]) ** 2
        return round(error,2)

# generate empty matrix (more faster than random)
n_item = 119867
n_user = 676436
big_matrix = np.zeros((n_user, n_item), dtype=np.float16)


def main():

    R = np.array([
        [5.0, 3, 0, 1],
        [4, 0, 0, 1],
        [1, 1, 0, 5],
        [1, 0, 0, 4],
        [0, 1, 5, 4],
    ], dtype=np.float128)

    matrix = MatrixFactorization(R, latent_factor=20, iterations=10, alpha=0.1)
    matrix.factorize()
    print("Final Predication\n", np.round(matrix.matrix_prediction()))
    # print(R.nonzero())

# main()
