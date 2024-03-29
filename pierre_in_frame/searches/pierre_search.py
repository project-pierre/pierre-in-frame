import itertools
import random
from pprint import pprint
from statistics import mean

import recommender_pierre
from datasets.registred_datasets import RegisteredDataset
from datasets.utils import split
from datasets.utils.split import SequentialTimeSplit
from scikit_pierre.metrics.evaluation import mean_average_precision
from searches.parameters import PierreParams
from settings.labels import Label
from settings.save_and_load import SaveAndLoad


class PierreGridSearch:

    def __init__(
            self,
            algorithm: str,
            dataset_name: str, n_splits: int = 3, trial: int = 1, fold: int = 3,
            n_jobs: int = 1, list_size: int = 10, n_inter: int = 50,
            based_on: str = "RANDOM"
    ):
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

    def fit_autoencoders(
            self, factors, epochs, dropout, lr, reg, train_list, test_list
    ):
        map_value = []

        for train, test in zip(train_list, test_list):
            if self.algorithm == Label.AUTOENC:
                recommender = recommender_pierre.AutoEncModel.AutoEncModel(
                    factors=int(factors), epochs=int(epochs), dropout=int(dropout), lr=int(lr), reg=int(reg),
                    batch=8
                )
            else:
                recommender = recommender_pierre.CDAEModel.CDAEModel(
                    factors=int(factors), epochs=int(epochs), dropout=int(dropout), lr=int(lr), reg=int(reg),
                    batch=8
                )
            rec_lists_df = recommender.train_and_produce_rec_list(user_transactions_df=train)
            map_value.append(mean_average_precision(rec_lists_df, test))

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

    def fit(self):
        """
        TODO: Docstring
        """
        print(self.algorithm)
        print(self.dataset.system_name)
        train_list = []
        test_list = []
        self.users_preferences = self.dataset.get_train_transactions(fold=self.fold, trial=self.trial)
        if self.based_on == Label.TIME:
            instance = SequentialTimeSplit(transactions_df=self.users_preferences, n_folds=self.n_splits)
            train_df, test_df = instance.main()
            train_list.append(train_df)
            test_list.append(test_df)
        else:
            cv_folds = split.split_with_joblib(transactions_df=self.users_preferences, trial=1, n_folds=self.n_splits)

            for train, test in cv_folds:
                train_list.append(train)
                test_list.append(test)

        output = []
        param_distributions = PierreParams.DAE_PARAMS
        combination = [
            param_distributions['factors'], param_distributions['epochs'],
            param_distributions['dropout'], param_distributions['lr'],
            param_distributions['reg']
        ]
        params_to_use = random.sample(list(itertools.product(*combination)), self.n_inter)
        # Starting the recommender algorithm
        # output = Parallel(n_jobs=self.n_jobs)(
        #     delayed(self.fit_autoencoders)(
        #         factors=factors, epochs=epochs, dropout=dropout, lr=lr,
        #         reg=reg, train_list=train_list, test_list=test_list
        #     ) for factors, epochs, dropout, lr, reg in params_to_use
        # )

        output = [
            self.fit_autoencoders(
                factors=factors, epochs=epochs, dropout=dropout, lr=lr,
                reg=reg, train_list=train_list, test_list=test_list
            ) for factors, epochs, dropout, lr, reg in params_to_use
        ]

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
