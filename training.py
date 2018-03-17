from sklearn.model_selection import train_test_split

from config import MODEL_LOCATION, DATASET_DIR
from recommender.fairness_reg_als import FairnessRegALS
from recommender.util import load_dataset, dataframe_to_matrix


def main():

    # load from previous model
    als = FairnessRegALS.load_data(MODEL_LOCATION)

    # create new if isn't available
    if als is None:

        # prepare dataset
        ratings_df = load_dataset(DATASET_DIR)
        train, test = train_test_split(ratings_df, test_size=0.2)
        print("total user dataset: {}, item dataset: {}"
              .format(ratings_df.user_id.unique().shape, ratings_df.item_id.unique().shape))
        print("total user training: {}, item training: {}"
              .format(train.user_id.unique().shape, train.item_id.unique().shape))
        print("total user test: {}, item test: {}"
              .format(test.user_id.unique().shape, test.item_id.unique().shape))
        print("total record training: {}, total record test: {}"
              .format(train.shape[0], test.shape[0]))

        # create new recommender instance
        als = FairnessRegALS(df_train=train, n_factor=50)

    # train the recommender
    als.train_data(iteration=30, directory=MODEL_LOCATION)
    als.save_data(MODEL_LOCATION)

if __name__ == '__main__':
    main()
