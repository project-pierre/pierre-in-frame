"""
Pierre in frame searches
"""
import itertools
import random
from pprint import pprint
from statistics import mean

import recommender_pierre
from datasets.registred_datasets import RegisteredDataset
from datasets.utils import split
from datasets.utils.split import SequentialTimeSplit
from scikit_pierre.metrics.evaluation import MeanAveragePrecision
from searches.parameters import PierreParams
from settings.labels import Label
from settings.save_and_load import SaveAndLoad


class PierreGridSearch:
    """
    Class for performing pierre grid search
    """

    def __init__(
            self,
            algorithm: str,
            dataset_name: str, n_splits: int = 3, trial: int = 1, fold: int = 3,
            n_jobs: int = 1, list_size: int = 10, n_inter: int = 50,
            based_on: str = "RANDOM"
    ):
        """
        Parameters
        """
        # global OPENBLAS_NUM_THREADS
        # OPENBLAS_NUM_THREADS = 1
        # threadpoolctl.threadpool_limits(1, "blas")
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

    def preparing_dataset(self):
        """
        Prepares the dataset for pierre search.
        """
        train_list = []
        test_list = []
        self.users_preferences = self.dataset.get_train_transactions(
            fold=self.fold, trial=self.trial
        )
        if self.based_on == Label.TIME:
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
        return train_list, test_list

    def fit_ease(self, lambda_: float, implicit: bool, train_list: list, test_list: list):
        """
        Fits the pierre grid search algorithm to the training set and testing set.
        """
        map_value = []

        for train, test in zip(train_list, test_list):
            recommender = recommender_pierre.EASEModel.EASEModel(
                lambda_=lambda_, implicit=implicit
            )
            map_value.append(self.__fit_and_metric(recommender, train, test))

        return {
            "map": mean(map_value),
            "params": {
                "lambda_": lambda_,
                "implicit": implicit
            }
        }

    def fit_autoencoders(
            self, factors, epochs, dropout, lr, reg, train_list, test_list
    ):
        """
        Fits the pierre grid search algorithm to the training set and testing set.
        """
        map_value = []

        for train, test in zip(train_list, test_list):
            if self.algorithm == Label.AUTOENC:
                recommender = recommender_pierre.AutoEncModel.AutoEncModel(
                    factors=int(factors), epochs=int(epochs), dropout=int(dropout), lr=int(lr),
                    reg=int(reg),
                    batch=8
                )
            else:
                recommender = recommender_pierre.CDAEModel.CDAEModel(
                    factors=int(factors), epochs=int(epochs), dropout=int(dropout), lr=int(lr),
                    reg=int(reg),
                    batch=8
                )
            map_value.append(self.__fit_and_metric(recommender, train, test))

        return {
            "map": mean(map_value),
            "params": {
                "factors": factors,
                "epochs": epochs,
                "dropout": dropout,
                "lr": lr,
                "reg": reg
            }
        }

    @staticmethod
    def __fit_and_metric(recommender, train, test):
        """
        Fits the pierre grid search algorithm to the training set and testing set.
        """
        rec_lists_df = recommender.train_and_produce_rec_list(user_transactions_df=train)
        metric_instance = MeanAveragePrecision(
            users_rec_list_df=rec_lists_df,
            users_test_set_df=test
        )
        return metric_instance.compute()

    def get_params_dae(self):
        """
        Returns the parameters of the pierre grid search algorithm.
        """
        param_distributions = PierreParams.DAE_PARAMS
        combination = [
            param_distributions['factors'], param_distributions['epochs'],
            param_distributions['dropout'], param_distributions['lr'],
            param_distributions['reg']
        ]
        params_to_use = random.sample(list(itertools.product(*combination)), self.n_inter)

        return params_to_use

    def get_params_ease(self):
        """
        Returns the parameters of the pierre grid search algorithm.
        """
        param_distributions = PierreParams.EASE_PARAMS
        combination = [
            param_distributions['lambda_'], param_distributions['implicit'],
        ]
        if self.n_inter < len(combination):
            params_to_use = random.sample(list(itertools.product(*combination)), self.n_inter)
        else:
            params_to_use = list(itertools.product(*combination))

        return params_to_use

    def defining_metric_and_save(self, output):
        """
        Saves the pierre grid search algorithm to a file.
        """
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

    def fit(self):
        """
        Fits the pierre grid search algorithm.
        """
        train_list, test_list = self.preparing_dataset()

        if self.algorithm in Label.ENCODERS_RECOMMENDERS:
            params_to_use = self.get_params_dae()
            output = [
                self.fit_autoencoders(
                    factors=factors, epochs=epochs, dropout=dropout, lr=lr,
                    reg=reg, train_list=train_list, test_list=test_list
                ) for factors, epochs, dropout, lr, reg in params_to_use
            ]
        else:
            params_to_use = self.get_params_ease()
            output = [
                self.fit_ease(
                    lambda_=lambda_, implicit=implicit, train_list=train_list, test_list=test_list
                ) for lambda_, implicit in params_to_use
            ]

        self.defining_metric_and_save(output=output)
