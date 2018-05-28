from sklearn.model_selection import train_test_split

from config import MODEL_LOCATION, DATASET_DIR
from recommender.fairness_reg_als import FairnessRegALS
from recommender.util import load_dataset, dataframe_to_matrix, divide_item_popularity

# Training Constant Parameter
N_FACTOR = 50
ITERATION = 1
LAMBDA_REG = 0.000001

# prepare dataset
RATINGS_DF = load_dataset(DATASET_DIR)


def main(session):

    model_location = MODEL_LOCATION + "-{}".format(session)

    # load from previous model
    als = FairnessRegALS.load_data(model_location)

    # create new if isn't available
    if als is None:

        train, test = train_test_split(RATINGS_DF, test_size=0.2)

        print("total user dataset: {}, item dataset: {}"
              .format(RATINGS_DF.user_id.unique().shape, RATINGS_DF.item_id.unique().shape))
        print("total user training: {}, item training: {}"
              .format(train.user_id.unique().shape, train.item_id.unique().shape))
        print("total user test: {}, item test: {}"
              .format(test.user_id.unique().shape, test.item_id.unique().shape))
        print("total record training: {}, total record test: {}"
              .format(train.shape[0], test.shape[0]))

        # create new recommender instance
        als = FairnessRegALS(df_train=train,
                             df_test=test,
                             n_factor=N_FACTOR,
                             lambda_reg=LAMBDA_REG * session)

    # train the recommender
    short_head, medium_tail = divide_item_popularity(RATINGS_DF)
    als.train_data(iteration=ITERATION, directory=model_location,
                   short_head=short_head, medium_tail=medium_tail)
    als.save_data(model_location)

if __name__ == '__main__':
    for session in range(1,10):
        main(session)
        print("session training: {} done\n".format(session))