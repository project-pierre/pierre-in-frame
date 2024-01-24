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
            if len(users_recommendation_lists) > 0:
                return True
            else:
                return False
        except Exception:
            return False

    @staticmethod
    def unit_step5_conformity_verification(
            cluster: str, recommender: str, dataset: str, trial: int, fold: int, metric: str,
            distribution: str, fairness: str, relevance: str, weight: str, tradeoff: str, selector: str
    ) -> bool:

        # Check integrity.
        try:
            metric_df = SaveAndLoad.load_conformity_metric(
                dataset=dataset, trial=trial, fold=fold,
                cluster=cluster, metric=metric, recommender=recommender,
                distribution=distribution, fairness=fairness, relevance=relevance,
                weight=weight, tradeoff=tradeoff, selector=selector
            )
            if len(metric_df[metric]) > 0:
                return True
            else:
                return False
        except Exception:
            return False

    @staticmethod
    def unit_step5_recommendation_verification(
            recommender: str, dataset: str, trial: int, fold: int, metric: str,
            distribution: str, fairness: str, relevance: str, weight: str, tradeoff: str, selector: str
    ) -> bool:

        # Check integrity.
        try:
            metric_df = SaveAndLoad.load_recommender_metric(
                metric=metric,
                recommender=recommender, dataset=dataset, trial=trial, fold=fold,
                distribution=distribution, fairness=fairness, relevance=relevance,
                weight=weight, tradeoff=tradeoff, selector=selector
            )
            if len(metric_df[metric]) > 0:
                return True
            else:
                return False
        except Exception:
            return False
