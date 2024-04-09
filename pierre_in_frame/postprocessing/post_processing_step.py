import logging

import pandas as pd

from datasets.candidate_items import CandidateItems
from datasets.registred_datasets import RegisteredDataset
from scikit_pierre.tradeoff.calibration import LinearCalibration, LogarithmBias
from settings.save_and_load import SaveAndLoad

logger = logging.getLogger(__name__)


class PostProcessingStep:
    """
    Class to lead with post-processing step
    """

    def __init__(
            self, recommender: str, dataset_name: str, fold: int, trial: int,
            tradeoff_component: str, distribution_component: str, fairness_component: str,
            relevance_component: str, tradeoff_weight_component: str, selector_component: str,
            list_size: int, alpha: int, d: int
    ):
        self.recommender = recommender
        self.fold = fold
        self.trial = trial
        self.tradeoff_component = tradeoff_component
        self.distribution_component = distribution_component
        self.fairness_component = fairness_component
        self.relevance_component = relevance_component
        self.tradeoff_weight_component = tradeoff_weight_component
        self.selector_component = selector_component
        self.list_size = list_size
        self.alpha = alpha
        self.d = d
        # Load dataset
        self.dataset = RegisteredDataset.load_dataset(dataset_name)
        # Load candidate items set
        self.users_distribution = None
        self.candidate_items = CandidateItems(recommender=recommender, dataset=dataset_name, trial=trial, fold=fold)
        try:
            self.users_distribution = SaveAndLoad.load_user_preference_distribution(
                dataset=dataset_name, fold=fold, trial=trial, distribution=distribution_component
            )
        except IOError:
            self.users_distribution = None
            print(IOError)
        # Choice the tradeoff
        if self.tradeoff_component == 'LIN':
            self.tradeoff_instance = LinearCalibration(
                users_preferences=self.dataset.get_train_transactions(fold=fold, trial=trial),
                candidate_items=self.candidate_items.get_candidate_items(),
                item_set=self.dataset.get_items(),
                users_distribution=self.users_distribution
            )
        elif self.tradeoff_component == 'LOG':
            self.tradeoff_instance = LogarithmBias(
                users_preferences=self.dataset.get_train_transactions(fold=fold, trial=trial),
                candidate_items=self.candidate_items.get_candidate_items(),
                item_set=self.dataset.get_items(),
                users_distribution=self.users_distribution
            )

        # Configuring the experimentation
        self.tradeoff_instance.config(
            distribution_component=distribution_component,
            fairness_component=fairness_component,
            relevance_component=relevance_component,
            tradeoff_weight_component=tradeoff_weight_component,
            select_item_component=selector_component,
            list_size=list_size, d=d, alpha=alpha
        )

    def run(self):
        """
        TODO: Docstring
        """

        # Execute the instance and get the recommendation list to all users.
        recommendation_lists = self.tradeoff_instance.fit()
        merged_results_df = pd.concat(recommendation_lists)

        # Save all recommendation lists
        SaveAndLoad.save_recommendation_lists(
            data=merged_results_df,
            recommender=self.recommender, dataset=self.dataset.system_name,
            trial=self.trial, fold=self.fold,
            tradeoff=self.tradeoff_component,
            distribution=self.distribution_component,
            fairness=self.fairness_component,
            relevance=self.relevance_component,
            tradeoff_weight=self.tradeoff_weight_component,
            select_item=self.selector_component
        )
