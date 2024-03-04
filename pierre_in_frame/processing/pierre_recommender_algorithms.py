import logging

import recommender_pierre
from datasets.registred_datasets import RegisteredDataset
from settings.labels import Label
from settings.save_and_load import SaveAndLoad

logger = logging.getLogger(__name__)


class PierreRecommenderAlgorithm:
    """
    Class to lead with the pierre recommender algorithms, generating the recommendation and saving in the results path
    """

    def __init__(self, recommender_name: str, dataset_name: str, fold: int, trial: int, list_size: int, metric: str = "map"):
        """
        Class constructor.

        :param recommender_name: The recommender algorithm name to be load and fit.
        :param dataset_name:  The dataset name to be used by the recommender algorithm.
        :param fold: The fold number to be load.
        :param trial: The trial number to be load.
        :param list_size: The recommendation list size.
        """
        self.dataset = RegisteredDataset.load_dataset(dataset_name)
        self.recommender_name = recommender_name
        self.fold = fold
        self.trial = trial
        self.recommender = None
        self.list_size = list_size

        # Load the surprise recommender algorithm
        full_params = SaveAndLoad.load_hyperparameters_recommender(
            dataset=self.dataset.system_name, algorithm=self.recommender_name,
            trial=self.trial, fold=self.fold
        )
        if self.recommender_name == Label.AUTOENC:
            self.recommender = recommender_pierre.AutoEncModel.AutoEncModel(
                factors=int(full_params["params"]["factors"]), epochs=int(full_params["params"]["epochs"]),
                dropout=int(full_params["params"]["dropout"]), lr=int(full_params["params"]["lr"]),
                reg=int(full_params["params"]["reg"]), list_size=int(self.list_size)
            )
        else:
            self.recommender = recommender_pierre.CDAEModel.CDAEModel(
                factors=int(full_params["params"]["factors"]), epochs=int(full_params["params"]["epochs"]),
                dropout=int(full_params["params"]["dropout"]), lr=int(full_params["params"]["lr"]),
                reg=int(full_params["params"]["reg"]), list_size=int(self.list_size)
            )

    def run(self):
        """
        Method to run the recommender algorithm, made and save the recommendation list
        """
        # fit the recommender algorithm
        logger.info(">>> Fit the recommender algorithm")
        users_preferences = self.dataset.get_train_transactions(fold=self.fold, trial=self.trial)

        rec_lists_df = self.recommender.train_and_produce_rec_list(user_transactions_df=users_preferences)

        # Save all recommendation lists
        logger.info(">>> Saving...")
        SaveAndLoad.save_candidate_items(
            data=rec_lists_df,
            dataset=self.dataset.system_name, algorithm=self.recommender_name,
            fold=self.fold, trial=self.trial
        )
