"""
Pierre in frame searches
"""
import itertools
import random
from statistics import mean

import recommender_pierre
from scikit_pierre.metrics.evaluation import MeanAveragePrecision
from searches.base_search import BaseSearch
from searches.parameters import PierreParams
from settings.labels import Label


class PierreGridSearch(BaseSearch):
    """
    Class for performing pierre grid search
    """

    def __init__(
            self,
            algorithm: str,
            dataset_name: str, trial: int = 1, fold: int = 3,
            n_jobs: int = 1, list_size: int = 10, n_inter: int = 50,
            based_on: str = "RANDOM"
    ):
        """
        Parameters
        """
        super().__init__(
            algorithm=algorithm, dataset_name=dataset_name, trial=trial, fold=fold,
            n_jobs=n_jobs, list_size=list_size, n_inter=n_inter, based_on=based_on
        )

    def fit_ease(self, lambda_: float, implicit: bool, train_list: list, valid_list: list):
        """
        Fits the pierre grid search algorithm to the training set and testing set.
        """
        map_value = []

        for train, test in zip(train_list, valid_list):
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
            self, factors, epochs, dropout, lr, reg, train_list, valid_list
    ):
        """
        Fits the pierre grid search algorithm to the training set and testing set.
        """
        map_value = []

        for train, test in zip(train_list, valid_list):
            if self.algorithm == Label.DEEP_AE:
                recommender = recommender_pierre.DeppAutoEncModel.DeppAutoEncModel(
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

        if self.n_inter < len(combination):
            params_to_use = random.sample(list(itertools.product(*combination)), self.n_inter)
        else:
            params_to_use = list(itertools.product(*combination))

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

    def preparing_recommenders(self):
        if self.algorithm in Label.ENCODERS_RECOMMENDERS:
            params_to_use = self.get_params_dae()
            self.output = [
                self.fit_autoencoders(
                    factors=factors, epochs=epochs, dropout=dropout, lr=lr,
                    reg=reg, train_list=self.train_list, valid_list=self.valid_list
                ) for factors, epochs, dropout, lr, reg in params_to_use
            ]
        else:
            params_to_use = self.get_params_ease()
            self.output = [
                self.fit_ease(
                    lambda_=lambda_, implicit=implicit,
                    train_list=self.train_list, valid_list=self.valid_list
                ) for lambda_, implicit in params_to_use
            ]
