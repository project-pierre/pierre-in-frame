import logging
from surprise import SVD, KNNBasic
from surprise.model_selection import RandomizedSearchCV
from surprise.prediction_algorithms.co_clustering import CoClustering
from surprise.prediction_algorithms.matrix_factorization import SVDpp, NMF

from datasets.registred_datasets import RegisteredDataset
from processing.conversions.pandas_surprise import PandasSurprise
from searches.parameters import SurpriseParams
from settings.constants import Constants
from settings.labels import Label
from settings.save_and_load import SaveAndLoad

logger = logging.getLogger(__name__)


class RecommenderSearch:
    """
    Class used to lead with the Random Search
    """

    def __init__(
            self, recommender: str, dataset: str, trial: int = None, fold: int = None,
            n_inter: int = Constants.N_INTER, n_jobs: int = Constants.N_CORES,
            n_cv: int = Constants.K_FOLDS_VALUE
    ):
        self.measures = ['rmse', 'mae', 'fcp', 'mse']
        self.trial = trial
        self.fold = fold
        self.n_inter = n_inter
        self.n_jobs = n_jobs
        self.n_cv = n_cv
        self.dataset = RegisteredDataset.load_dataset(dataset)
        self.recommender_name = recommender
        self.recommender = None
        self.params = None
        if recommender == Label.SVD:
            self.recommender = SVD
            self.params = SurpriseParams.SVD_SEARCH_PARAMS
        elif recommender == Label.NMF:
            self.recommender = NMF
            self.params = SurpriseParams.NMF_SEARCH_PARAMS
        elif recommender == Label.CO_CLUSTERING:
            self.recommender = CoClustering
            self.params = SurpriseParams.CLUSTERING_SEARCH_PARAMS
        elif recommender == Label.ITEM_KNN_BASIC:
            self.recommender = KNNBasic
            self.params = SurpriseParams.ITEM_KNN_SEARCH_PARAMS
        elif recommender == Label.USER_KNN_BASIC:
            self.recommender = KNNBasic
            self.params = SurpriseParams.USER_KNN_SEARCH_PARAMS
        else:
            self.recommender = SVDpp
            self.params = SurpriseParams.SVDpp_SEARCH_PARAMS

    def __search(self):
        """
        Randomized Search Cross Validation to get the best params in the recommender algorithm
        :return: A Random Search instance
        """
        gs = RandomizedSearchCV(
            algo_class=self.recommender, param_distributions=self.params, measures=self.measures,
            n_iter=self.n_inter, cv=self.n_cv,
            n_jobs=self.n_jobs, joblib_verbose=100, random_state=42
        )
        gs.fit(
            PandasSurprise.pandas_transform_all_dataset_to_surprise(
                self.dataset.get_full_train_transactions(trial=self.trial, fold=self.fold)
            )
        )
        return gs

    def fit(self) -> None:
        """
        Search and save the best param values
        """
        gs = self.__search()
        # Saving
        SaveAndLoad.save_hyperparameters_recommender(
            best_params=gs.best_params,
            dataset=self.dataset.system_name, algorithm=self.recommender_name
        )
