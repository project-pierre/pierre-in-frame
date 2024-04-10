import implicit
import itertools
import pandas as pd
import random
import threadpoolctl
from joblib import Parallel, delayed
from pprint import pprint
from scipy import sparse
from statistics import mean

from datasets.registred_datasets import RegisteredDataset
from datasets.utils import split
from datasets.utils.split import SequentialTimeSplit
from scikit_pierre.metrics.evaluation import MeanAveragePrecision
from searches.parameters import ImplicitParams
from settings.labels import Label
from settings.save_and_load import SaveAndLoad


class ImplicitGridSearch:

    def __init__(
            self,
            algorithm: str,
            dataset_name: str, n_splits: int = 3, trial: int = 1, fold: int = 3,
            n_jobs: int = 1, list_size: int = 10, n_inter: int = 50,
            based_on: str = "RANDOM"
    ):
        global OPENBLAS_NUM_THREADS
        OPENBLAS_NUM_THREADS = 1
        threadpoolctl.threadpool_limits(1, "blas")
        self.dataset = RegisteredDataset.load_dataset(dataset_name)
        self.algorithm = algorithm
        self.trial = trial
        self.fold = fold
        self.n_splits = n_splits
        self.n_inter = n_inter
        self.n_jobs = n_jobs
        self.list_size = list_size
        self.users_preferences = None
        self.based_on = based_on

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
            train_list, test_list
    ):
        map_value = []

        for train, test in zip(train_list, test_list):
            recommender = implicit.als.AlternatingLeastSquares(
                factors=factors, regularization=regularization, alpha=alpha, iterations=iterations,
                random_state=random_state, num_threads=1
            )
            rec_lists_df = self.__run__(recommender=recommender, users_preferences=train)
            metric_instance = MeanAveragePrecision(
                users_rec_list_df=rec_lists_df,
                users_test_set_df=test
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
            train_list, test_list
    ):
        map_value = []

        for train, test in zip(train_list, test_list):
            recommender = implicit.bpr.BayesianPersonalizedRanking(
                factors=factors, regularization=regularization, learning_rate=learning_rate,
                iterations=iterations, random_state=random_state, num_threads=1
            )
            rec_lists_df = self.__run__(recommender=recommender, users_preferences=train)
            metric_instance = MeanAveragePrecision(
                users_rec_list_df=rec_lists_df,
                users_test_set_df=test
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

    def fit(self):
        """
        TODO: Docstring
        """
        print(self.algorithm)
        print(self.dataset.system_name)
        train_list = []
        test_list = []
        self.users_preferences = self.dataset.get_train_transactions(
            fold=self.fold, trial=self.trial
        )
        if self.based_on == Label.TIME:
            cv_folds = []
            instance = SequentialTimeSplit(
                transactions_df=self.users_preferences, n_folds=self.n_splits
            )
            train_df, test_df = instance.main()
            train_list.append(train_df)
            test_list.append(test_df)
        else:
            cv_folds = split.split_with_joblib(
                transactions_df=self.users_preferences, trial=1, n_folds=self.n_splits
            )

            for train, test in cv_folds:
                train_list.append(train)
                test_list.append(test)

        output = []
        if self.algorithm == Label.ALS:
            param_distributions = ImplicitParams.ALS_PARAMS
            combination = [
                param_distributions['factors'], param_distributions['regularization'],
                param_distributions['alpha'], param_distributions['iterations'],
                param_distributions['random_state'], param_distributions['num_threads'],
            ]
            params_to_use = random.sample(list(itertools.product(*combination)), self.n_inter)
            # Starting the recommender algorithm
            output = Parallel(n_jobs=self.n_jobs)(
                delayed(self.fit_als)(
                    factors=factors, regularization=regularization, alpha=alpha,
                    iterations=iterations,
                    random_state=random_state, num_threads=num_threads, train_list=train_list,
                    test_list=test_list
                ) for factors, regularization, alpha, iterations, random_state, num_threads in params_to_use
            )
        elif self.algorithm == Label.BPR:
            param_distributions = ImplicitParams.BPR_PARAMS
            combination = [
                param_distributions['factors'], param_distributions['regularization'],
                param_distributions['learning_rate'], param_distributions['iterations'],
                param_distributions['random_state'], param_distributions['num_threads'],
            ]

            params_to_use = random.sample(list(itertools.product(*combination)), self.n_inter)
            # Starting the recommender algorithm
            output = Parallel(n_jobs=self.n_jobs)(
                delayed(self.fit_bpr)(
                    factors=factors, regularization=regularization, learning_rate=learning_rate, iterations=iterations,
                    random_state=random_state, num_threads=num_threads, train_list=train_list, test_list=test_list
                ) for factors, regularization, learning_rate, iterations, random_state, num_threads in params_to_use
            )
        elif self.algorithm == Label.LMF:
            pass
        else:
            pass
        best_params = {
            "map": 0.0
        }
        for item in output:
            if float(best_params["map"]) < float(item["map"]):
                best_params = item
        pprint(best_params)
        # Saving
        SaveAndLoad.save_hyperparameters_recommender(
            best_params=best_params, dataset=self.dataset.system_name, algorithm=self.algorithm,
            trial=self.trial, fold=self.fold
        )
