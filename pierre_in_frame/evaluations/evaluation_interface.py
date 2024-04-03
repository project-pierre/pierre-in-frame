from collections import Counter

import logging
import pandas as pd
import recmetrics

from checkpoint_verification import CheckpointVerification
from datasets.registred_datasets import RegisteredDataset
from scikit_pierre.metrics.evaluation import (
    Miscalibration, MeanAbsoluteCalibrationError,
    MeanAveragePrecision, MeanReciprocalRank, MeanAverageMiscalibration,
    AverageNumberOfOItemsChanges, AverageNumberOfGenreChanges, Unexpectedness, Serendipity,
    ExplainingMiscalibration, NumberOfUserIncreaseAndDecreaseMiscalibration, UserIDMiscalibration,
    IntraListSimilarity
)
from settings.constants import Constants
from settings.labels import Label
from settings.path_dir_file import PathDirFile
from settings.save_and_load import SaveAndLoad

logger = logging.getLogger(__name__)


class ApplyingMetric:
    """

    """

    def __init__(
            self,
            recommender: str, dataset: str, trial: int, fold: int,
            distribution: str, fairness: str, relevance: str,
            weight: str, tradeoff: str, selector: str,
            checkpoint: str
    ):
        """

        """
        self.recommender = recommender
        self.dataset = dataset
        self.trial = trial
        self.fold = fold
        self.distribution = distribution
        self.fairness = fairness
        self.relevance = relevance
        self.weight = weight
        self.tradeoff = tradeoff
        self.selector = selector
        self.checkpoint = checkpoint
        self.metric = None
        self.users_prof_df = None
        self.users_rec_list_df = None
        self.users_test_set_df = None
        self.users_cand_items_df = None
        self.users_baseline_df = None
        self.dataset_instance = None
        self.metric_instance = None
        self.items_set = None

    def set_metric(self, metric: str):
        """

        """
        self.metric = metric

    def load_user_profile(self):
        self.users_prof_df = self.dataset_instance.get_train_transactions(
            trial=self.trial, fold=self.fold
        )

    def load_rec_list(self):
        """

        """
        self.users_rec_list_df = SaveAndLoad.load_recommendation_lists(
            recommender=self.recommender, dataset=self.dataset, trial=self.trial, fold=self.fold,
            distribution=self.distribution, fairness=self.fairness, relevance=self.relevance,
            tradeoff_weight=self.weight, tradeoff=self.tradeoff, select_item=self.selector
        )

    def load_rec_baseline(self):
        """

        """
        if self.users_baseline_df is None:
            self.users_baseline_df = SaveAndLoad.load_recommendation_lists(
                recommender=self.recommender, dataset=self.dataset, trial=self.trial, fold=self.fold,
                distribution=self.distribution, fairness=self.fairness, relevance=self.relevance,
                tradeoff_weight="C@0.0", tradeoff=self.tradeoff, select_item=self.selector
            )

    def load_items_set(self):
        self.items_set = self.dataset_instance.get_items()

    def load_candidate_items(self):
        """

        """
        self.users_cand_items_df = SaveAndLoad.load_candidate_items(
            algorithm=self.recommender, dataset=self.dataset, trial=self.trial, fold=self.fold
        )

    def load_test_set(self):
        self.users_test_set_df = self.dataset_instance.get_test_transactions(
            trial=self.trial, fold=self.fold
        )

    def load_dataset(self):
        """

        """
        self.dataset_instance = RegisteredDataset.load_dataset(self.dataset)

    def load(self):
        self.load_dataset()
        self.load_rec_list()
        self.load_test_set()
        self.load_candidate_items()
        self.load_items_set()
        self.load_user_profile()

    def load_anc(self):
        self.metric_instance = AverageNumberOfOItemsChanges(
            users_rec_list_df=self.users_rec_list_df,
            users_baseline_df=self.users_baseline_df
        )

    def load_angc(self):
        self.metric_instance = AverageNumberOfGenreChanges(
            users_baseline_df=self.users_baseline_df,
            users_rec_list_df=self.users_rec_list_df,
            items_df=self.items_set
        )

    def load_map(self):
        self.metric_instance = MeanAveragePrecision(
            users_rec_list_df=self.users_rec_list_df,
            users_test_set_df=self.users_test_set_df
        )

    def load_mrr(self):
        self.metric_instance = MeanReciprocalRank(
            users_rec_list_df=self.users_rec_list_df,
            users_test_set_df=self.users_test_set_df
        )

    def load_unexpectedness(self):
        self.metric_instance = Unexpectedness(
            users_rec_list_df=self.users_rec_list_df,
            users_test_df=self.users_test_set_df
        )

    def load_ils(self):
        self.metric_instance = IntraListSimilarity(
            users_rec_list_df=self.users_rec_list_df,
            items_df=self.items_set
        )

    def load_serendipity(self):
        self.metric_instance = Serendipity(
            users_rec_list_df=self.users_rec_list_df,
            users_test_df=self.users_test_set_df,
            users_baseline_df=self.users_baseline_df
        )

    def load_mace(self):
        self.metric_instance = MeanAbsoluteCalibrationError(
            users_profile_df=self.users_prof_df,
            users_rec_list_df=self.users_rec_list_df,
            items_set_df=self.items_set,
            distribution_name=self.distribution
        )

    def load_mc(self):
        self.metric_instance = Miscalibration(
            users_profile_df=self.users_prof_df,
            users_rec_list_df=self.users_rec_list_df,
            items_set_df=self.items_set,
            distribution_name=self.distribution,
            distance_func_name=self.fairness
        )

    def load_mamc(self):
        self.metric_instance = MeanAverageMiscalibration(
            users_profile_df=self.users_prof_df,
            users_rec_list_df=self.users_rec_list_df,
            items_set_df=self.items_set,
            distribution_name=self.distribution,
            distance_func_name=self.fairness
        )

    def load_exp_mc(self):
        self.metric_instance = ExplainingMiscalibration(
            users_profile_df=self.users_prof_df,
            users_rec_list_df=self.users_rec_list_df,
            users_baseline_df=self.users_baseline_df,
            items_df=self.items_set,
            distribution_name=self.distribution,
            distance_func_name=self.fairness
        )

    def load_inc_dec_mc(self, increase: bool):
        self.metric_instance = NumberOfUserIncreaseAndDecreaseMiscalibration(
            users_profile_df=self.users_prof_df,
            users_rec_list_df=self.users_rec_list_df,
            users_baseline_df=self.users_baseline_df,
            items_df=self.items_set,
            distribution_name=self.distribution,
            distance_func_name=self.fairness
        )
        self.metric_instance.set_choice(choice=increase)

    def load_user_inc_dec_mc(self, increase: bool):
        self.metric_instance = UserIDMiscalibration(
            users_profile_df=self.users_prof_df,
            users_rec_list_df=self.users_rec_list_df,
            users_baseline_df=self.users_baseline_df,
            items_df=self.items_set,
            distribution_name=self.distribution,
            distance_func_name=self.fairness
        )
        self.metric_instance.set_choice(choice=increase)

    def compute(self):
        _value = self.metric_instance.compute()

        results = pd.DataFrame([[
            _value
        ]], columns=[self.metric])

        self.saving_results(results)

    def saving_results(self, results):
        SaveAndLoad.save_recommender_metric(
            data=results, metric=self.metric,
            recommender=self.recommender, dataset=self.dataset, trial=self.trial, fold=self.fold,
            distribution=self.distribution, fairness=self.fairness, relevance=self.relevance,
            weight=self.weight, tradeoff=self.tradeoff, selector=self.selector
        )

    def verifying_checkpoint(self):
        if (
            self.checkpoint == "YES" and
            CheckpointVerification.unit_step5_recommendation_verification(
                dataset=self.dataset, trial=self.trial, fold=self.fold,
                metric=self.metric, recommender=self.recommender,
                distribution=self.distribution, fairness=self.fairness,
                relevance=self.relevance,
                weight=self.weight, tradeoff=self.tradeoff, selector=self.selector
            )
        ):
            return True
        return False


def applying_novelty(
        recommender: str, dataset: str, trial: int, fold: int,
        distribution: str, fairness: str, relevance: str, weight: str, tradeoff: str, selector: str,
        checkpoint: str
):
    """
    Function that apply the evaluation metrics.
    """

    system_name = "-".join([
        dataset, 'trial-' + str(trial), 'fold-' + str(fold), recommender,
        tradeoff, distribution, relevance, selector, fairness, tradeoff, "-->", Label.NOVELTY
    ])

    if checkpoint == "YES" and CheckpointVerification.unit_step5_recommendation_verification(
            dataset=dataset, trial=trial, fold=fold,
            metric=Label.NOVELTY, recommender=recommender,
            distribution=distribution, fairness=fairness, relevance=relevance,
            weight=weight, tradeoff=tradeoff, selector=selector
    ):
        logger.info(">> Already Done... " + system_name)
        return "Already Done"

    dataset_instance = RegisteredDataset.load_dataset(dataset)
    train_df = dataset_instance.get_transactions()

    path = PathDirFile.get_recommendation_list_file(
        dataset=dataset, recommender=recommender, trial=trial, fold=fold,
        tradeoff=tradeoff, distribution=distribution, fairness=fairness,
        relevance=relevance, tradeoff_weight=weight, select_item=selector
    )
    users_recommendation_lists = pd.read_csv(path)

    # Executing

    rec_set = [row[Label.ITEM_ID].tolist() for ix, row in
               users_recommendation_lists.groupby(by=[Label.USER_ID])]
    pop = Counter(train_df[Label.ITEM_ID].tolist())
    u = train_df[Label.USER_ID].nunique()
    metric_value, _ = recmetrics.novelty(
        predicted=rec_set, pop=pop, u=u, n=Constants.RECOMMENDATION_LIST_SIZE
    )

    results = pd.DataFrame([[
        metric_value
    ]], columns=[Label.NOVELTY])

    SaveAndLoad.save_recommender_metric(
        data=results,
        recommender=recommender, dataset=dataset, trial=trial, fold=fold,
        distribution=distribution, fairness=fairness, relevance=relevance,
        weight=weight, tradeoff=tradeoff, selector=selector,
        metric=Label.NOVELTY
    )

    return "Finished"


def applying_coverage(
        recommender: str, dataset: str, trial: int, fold: int,
        distribution: str, fairness: str, relevance: str, weight: str, tradeoff: str, selector: str,
        checkpoint: str
):
    """
    Function that apply the evaluation metrics.
    """

    system_name = "-".join([
        dataset, 'trial-' + str(trial), 'fold-' + str(fold), recommender,
        tradeoff, distribution, relevance, selector, fairness, tradeoff, "-->", Label.COVERAGE
    ])

    if checkpoint == "YES" and CheckpointVerification.unit_step5_recommendation_verification(
            dataset=dataset, trial=trial, fold=fold,
            metric=Label.COVERAGE, recommender=recommender,
            distribution=distribution, fairness=fairness, relevance=relevance,
            weight=weight, tradeoff=tradeoff, selector=selector
    ):
        logger.info(">> Already Done... " + system_name)
        return "Already Done"

    dataset_instance = RegisteredDataset.load_dataset(dataset)
    items_df = dataset_instance.get_items()

    path = PathDirFile.get_recommendation_list_file(
        dataset=dataset, recommender=recommender, trial=trial, fold=fold,
        tradeoff=tradeoff, distribution=distribution, fairness=fairness,
        relevance=relevance, tradeoff_weight=weight, select_item=selector
    )
    users_recommendation_lists = pd.read_csv(path)
    rec_set = [row[Label.ITEM_ID].tolist() for ix, row in
               users_recommendation_lists.groupby(by=[Label.USER_ID])]

    # Executing
    metric_value = recmetrics.prediction_coverage(
        predicted=rec_set, catalog=items_df[Label.ITEM_ID].tolist()
    )

    results = pd.DataFrame([[
        metric_value
    ]], columns=[Label.COVERAGE])

    SaveAndLoad.save_recommender_metric(
        data=results,
        recommender=recommender, dataset=dataset, trial=trial, fold=fold,
        distribution=distribution, fairness=fairness, relevance=relevance,
        weight=weight, tradeoff=tradeoff, selector=selector,
        metric=Label.COVERAGE
    )

    return "Finished"


def applying_personalization(
        recommender: str, dataset: str, trial: int, fold: int,
        distribution: str, fairness: str, relevance: str, weight: str, tradeoff: str, selector: str,
        checkpoint: str
):
    """
    Function that apply the evaluation metrics.
    """

    system_name = "-".join([
        dataset, 'trial-' + str(trial), 'fold-' + str(fold), recommender,
        tradeoff, distribution, relevance, selector, fairness, tradeoff, "-->",
        Label.PERSONALIZATION
    ])

    if checkpoint == "YES" and CheckpointVerification.unit_step5_recommendation_verification(
            dataset=dataset, trial=trial, fold=fold,
            metric=Label.PERSONALIZATION, recommender=recommender,
            distribution=distribution, fairness=fairness, relevance=relevance,
            weight=weight, tradeoff=tradeoff, selector=selector
    ):
        logger.info(">> Already Done... " + system_name)
        return "Already Done"

    path = PathDirFile.get_recommendation_list_file(
        dataset=dataset, recommender=recommender, trial=trial, fold=fold,
        tradeoff=tradeoff, distribution=distribution, fairness=fairness,
        relevance=relevance, tradeoff_weight=weight, select_item=selector
    )
    users_recommendation_lists = pd.read_csv(path)
    rec_set = [row[Label.ITEM_ID].tolist() for ix, row in
               users_recommendation_lists.groupby(by=[Label.USER_ID])]

    # Executing
    metric_value = recmetrics.personalization(
        predicted=rec_set
    )

    results = pd.DataFrame([[
        metric_value
    ]], columns=[Label.PERSONALIZATION])

    SaveAndLoad.save_recommender_metric(
        data=results,
        recommender=recommender, dataset=dataset, trial=trial, fold=fold,
        distribution=distribution, fairness=fairness, relevance=relevance,
        weight=weight, tradeoff=tradeoff, selector=selector,
        metric=Label.PERSONALIZATION
    )

    return "Finished"


def execution_time_fold(
        recommender: str, dataset: str, trial: int, fold: int,
        distribution: str, fairness: str, relevance: str, weight: str, tradeoff: str, selector: str,
        checkpoint: str
):
    path = PathDirFile.get_postprocessing_time_file(
        dataset=dataset, recommender=recommender, trial=trial, fold=fold,
        tradeoff=tradeoff, distribution=distribution, fairness=fairness,
        relevance=relevance, tradeoff_weight=weight, select_item=selector)
    execution_time_df = pd.read_csv(path)
    time_values = execution_time_df['finished_at'] - execution_time_df['stated_at']

    results = pd.DataFrame([[
        time_values
    ]], columns=['TIME'])

    SaveAndLoad.save_recommender_metric(
        data=results, metric="TIME",
        recommender=recommender, dataset=dataset, trial=trial, fold=fold,
        distribution=distribution, fairness=fairness, relevance=relevance,
        weight=weight, tradeoff=tradeoff, selector=selector
    )
    return "Finished"
