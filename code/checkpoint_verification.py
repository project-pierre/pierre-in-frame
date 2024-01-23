import logging

from settings.save_and_load import SaveAndLoad

logger = logging.getLogger(__name__)


class CheckpointVerification:
    def __init__(self):
        pass

    @staticmethod
    def unit_step4_verification(
        dataset: str, recommender: str, trial: int, fold: int,
        tradeoff: str, distribution: str, fairness: str, relevance: str,
        tradeoff_weight: str, select_item: str
    ) -> bool:

        # Check integrity.
        try:
            users_recommendation_lists = SaveAndLoad.load_recommendation_lists(
                dataset=dataset, recommender=recommender, trial=trial, fold=fold,
                tradeoff=tradeoff, distribution=distribution, fairness=fairness,
                relevance=relevance, tradeoff_weight=tradeoff_weight, select_item=select_item
            )
            if len(users_recommendation_lists) > 100:
                return True
            else:
                return False
        except Exception as e:
            return False
