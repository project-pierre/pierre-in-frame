import implicit
import itertools
import pandas as pd
import random
import threadpoolctl
from joblib import Parallel, delayed
from scipy import sparse
from statistics import mean

from scikit_pierre.metrics.evaluation import MeanAveragePrecision
from searches.base_search import BaseSearch
from searches.parameters import ImplicitParams
from settings.labels import Label


class ImplicitGridSearch(BaseSearch):

    def __init__(
            self,
            algorithm: str,
            dataset_name: str, trial: int = 1, fold: int = 1,
            n_jobs: int = 1, list_size: int = 10, n_inter: int = 50,
            based_on: str = "RANDOM"
    ):
        global OPENBLAS_NUM_THREADS
        OPENBLAS_NUM_THREADS = 1
        threadpoolctl.threadpool_limits(1, "blas")
        super().__init__(
            algorithm=algorithm, dataset_name=dataset_name, trial=trial, fold=fold,
            n_jobs=n_jobs, list_size=list_size, n_inter=n_inter, based_on=based_on
        )

    def __predict(self, user_preferences: pd.DataFrame, user_id, recommender) -> pd.DataFrame:
        """
        Method to predict the rating to a user.

        :param user_preferences: A Pandas Dataframe with the user_id and item_id.
        :return: A Pandas Dataframe with the user_id, item_id and predicted_rating.
        """

        ids, scores = recommender.recommend(
            user_id, user_preferences, N=self.list_size, filter_already_liked_items=True
        )
        df = pd.DataFrame([], columns=[Label.USER_ID, Label.ITEM_ID, Label.TRANSACTION_VALUE])
        df[Label.ITEM_ID] = ids.tolist()
        df[Label.TRANSACTION_VALUE] = scores.tolist()
        df[Label.USER_ID] = user_id
        return df

    def __run__(self, recommender, users_preferences):
        """
        Method to run the recommender algorithm, made and save the recommendation list
        """
        # fit the recommender algorithm
        sparse_customer_item = sparse.csr_matrix(
            (
                users_preferences[Label.TRANSACTION_VALUE].astype(float),
                (users_preferences[Label.USER_ID], users_preferences[Label.ITEM_ID]),
            )
        )

        recommender.fit(sparse_customer_item)

        user_list = users_preferences[Label.USER_ID].unique()

        # Predict the recommendation list
        result_list = [self.__predict(
            user_preferences=sparse_customer_item[user_id],
            user_id=user_id,
            recommender=recommender
        ) for user_id in user_list]
        return pd.concat(result_list)

    def fit_als(
            self, factors, regularization, alpha, iterations, random_state, num_threads,
            train_list, valid_list
    ):
        map_value = []

        for train, validation in zip(train_list, valid_list):
            recommender = implicit.als.AlternatingLeastSquares(
                factors=factors, regularization=regularization, alpha=alpha, iterations=iterations,
                random_state=random_state, num_threads=1
            )
            rec_lists_df = self.__run__(recommender=recommender, users_preferences=train)
            metric_instance = MeanAveragePrecision(
                users_rec_list_df=rec_lists_df,
                users_test_set_df=validation
            )
            map_value.append(metric_instance.compute())

        return {
            "map": mean(map_value),
            "params": {
                "factors": factors,
                "regularization": regularization,
                "alpha": alpha,
                "iterations": iterations,
                "random_state": random_state
            }
        }

    def fit_bpr(
            self, factors, regularization, learning_rate, iterations, random_state, num_threads,
            train_list, valid_list
    ):
        map_value = []

        for train, validation in zip(train_list, valid_list):
            recommender = implicit.bpr.BayesianPersonalizedRanking(
                factors=factors, regularization=regularization, learning_rate=learning_rate,
                iterations=iterations, random_state=random_state, num_threads=1
            )
            rec_lists_df = self.__run__(recommender=recommender, users_preferences=train)
            metric_instance = MeanAveragePrecision(
                users_rec_list_df=rec_lists_df,
                users_test_set_df=validation
            )
            map_value.append(metric_instance.compute())

        return {
            "map": mean(map_value),
            "params": {
                "factors": factors,
                "regularization": regularization,
                "learning_rate": learning_rate,
                "iterations": iterations,
                "random_state": random_state
            }
        }

    def get_als_params(self):
        param_distributions = ImplicitParams.ALS_PARAMS

        combination = [
            param_distributions['factors'], param_distributions['regularization'],
            param_distributions['alpha'], param_distributions['iterations'],
            param_distributions['random_state'], param_distributions['num_threads'],
        ]
        if self.n_inter < len(combination):
            params_to_use = random.sample(list(itertools.product(*combination)), self.n_inter)
        else:
            params_to_use = list(itertools.product(*combination))
        return params_to_use

    def get_bpr_params(self):
        param_distributions = ImplicitParams.BPR_PARAMS

        combination = [
            param_distributions['factors'], param_distributions['regularization'],
            param_distributions['learning_rate'], param_distributions['iterations'],
            param_distributions['random_state'], param_distributions['num_threads'],
        ]
        if self.n_inter < len(combination):
            params_to_use = random.sample(list(itertools.product(*combination)), self.n_inter)
        else:
            params_to_use = list(itertools.product(*combination))
        return params_to_use

    def preparing_recommenders(self):
        if self.algorithm == Label.ALS:
            params_to_use = self.get_als_params()

            # Starting the recommender algorithm
            self.output = list(Parallel(n_jobs=self.n_jobs)(
                delayed(self.fit_als)(
                    factors=factors, regularization=regularization, alpha=alpha,
                    iterations=iterations,
                    random_state=random_state, num_threads=num_threads, train_list=self.train_list,
                    valid_list=self.valid_list
                ) for factors, regularization, alpha, iterations, random_state, num_threads in
                params_to_use
            ))
        elif self.algorithm == Label.BPR:
            params_to_use = self.get_bpr_params()

            # Starting the recommender algorithm
            self.output = list(Parallel(n_jobs=self.n_jobs)(
                delayed(self.fit_bpr)(
                    factors=factors, regularization=regularization,
                    learning_rate=learning_rate, iterations=iterations,
                    random_state=random_state, num_threads=num_threads,
                    train_list=self.train_list, valid_list=self.valid_list
                ) for factors, regularization, learning_rate, iterations, random_state, num_threads
                in params_to_use
            ))
        else:
            pass
