from collections import Counter

import os

import logging

import pandas as pd
import recmetrics

from datasets.registred_datasets import RegisteredDataset
from scikit_pierre.classes.genre import genre_probability_approach
from scikit_pierre.distributions.accessible import distributions_funcs_pandas
from scikit_pierre.measures.accessible import calibration_measures_funcs
from scikit_pierre.metrics.mace import mace
from scikit_pierre.metrics.map import mean_average_precision_map
from scikit_pierre.metrics.mrmc import mrmc
from scikit_pierre.metrics.mrr import mean_reciprocal_rank_map
from scikit_pierre.metrics.serendipity import serendipity
from scikit_pierre.metrics.unexpectedness import unexpectedness
from settings.constants import Constants
from settings.labels import Label
from settings.path_dir_file import PathDirFile
from settings.save_and_load import SaveAndLoad

logger = logging.getLogger(__name__)


def applying_mrr(recommender, dataset, trial, fold, distribution, fairness, relevance, weight, tradeoff, selector):
    """
    Function that apply the evaluation metrics.
    """
    try:
        metric_file = SaveAndLoad.load_recommender_metric(
            metric='MRR',
            recommender=recommender, dataset=dataset, trial=trial, fold=fold,
            distribution=distribution, fairness=fairness, relevance=relevance,
            weight=weight, tradeoff=tradeoff, selector=selector
        )
        # Check integrity
        if len(metric_file) == 1:
            return "AlreadyDone"
    except Exception as e:
        logger.error(" - ".join([str(e)]))

    dataset_instance = RegisteredDataset.load_dataset(dataset)

    path = PathDirFile.get_recommendation_list_file(
        dataset=dataset, recommender=recommender, trial=trial, fold=fold,
        tradeoff=tradeoff, distribution=distribution, fairness=fairness,
        relevance=relevance, tradeoff_weight=weight, select_item=selector
    )
    users_recommendation_lists = pd.read_csv(path)

    # Executing
    users_test_items = dataset_instance.get_test_transactions(trial=trial, fold=fold)
    mrr_value = mean_reciprocal_rank_map(users_recommendation_lists, users_test_items)

    results = pd.DataFrame([[
        mrr_value
    ]], columns=['MRR'])

    SaveAndLoad.save_recommender_metric(
        data=results, metric='MRR',
        recommender=recommender, dataset=dataset, trial=trial, fold=fold,
        distribution=distribution, fairness=fairness, relevance=relevance,
        weight=weight, tradeoff=tradeoff, selector=selector
    )
    return "Finished"


def applying_map(recommender, dataset, trial, fold, distribution, fairness, relevance, weight, tradeoff, selector):
    """
    Function that apply the evaluation metrics.
    """
    try:
        metric_file = SaveAndLoad.load_recommender_metric(
            metric='MAP',
            recommender=recommender, dataset=dataset, trial=trial, fold=fold,
            distribution=distribution, fairness=fairness, relevance=relevance,
            weight=weight, tradeoff=tradeoff, selector=selector
        )
        # Check integrity
        if len(metric_file) == 1:
            return "AlreadyDone"
    except Exception as e:
        logger.error(" - ".join([str(e)]))

    dataset_instance = RegisteredDataset.load_dataset(dataset)

    path = PathDirFile.get_recommendation_list_file(
        dataset=dataset, recommender=recommender, trial=trial, fold=fold,
        tradeoff=tradeoff, distribution=distribution, fairness=fairness,
        relevance=relevance, tradeoff_weight=weight, select_item=selector
    )
    users_recommendation_lists = pd.read_csv(path)

    # Executing
    users_test_items = dataset_instance.get_test_transactions(trial=trial, fold=fold)
    map_value = mean_average_precision_map(users_recommendation_lists, users_test_items)

    results = pd.DataFrame([[
        map_value
    ]], columns=['MAP'])

    SaveAndLoad.save_recommender_metric(
        data=results, metric="MAP",
        recommender=recommender, dataset=dataset, trial=trial, fold=fold,
        distribution=distribution, fairness=fairness, relevance=relevance,
        weight=weight, tradeoff=tradeoff, selector=selector
    )
    return "Finished"


def applying_mace(recommender, dataset, trial, fold, distribution, fairness, relevance, weight, tradeoff, selector):
    """
    Function that apply the evaluation metrics.
    """
    try:
        metric_file = SaveAndLoad.load_recommender_metric(
            metric="MACE",
            recommender=recommender, dataset=dataset, trial=trial, fold=fold,
            distribution=distribution, fairness=fairness, relevance=relevance,
            weight=weight, tradeoff=tradeoff, selector=selector
        )
        # Check integrity
        if len(metric_file) == 1:
            return "AlreadyDone"
    except Exception as e:
        logger.error(" - ".join([str(e)]))

    dataset_instance = RegisteredDataset.load_dataset(dataset)

    path = PathDirFile.get_recommendation_list_file(
        dataset=dataset, recommender=recommender, trial=trial, fold=fold,
        tradeoff=tradeoff, distribution=distribution, fairness=fairness,
        relevance=relevance, tradeoff_weight=weight, select_item=selector
    )
    users_recommendation_lists = pd.read_csv(path)

    # Executing
    items_set = dataset_instance.get_items()
    items_classes_set = genre_probability_approach(item_set=items_set)

    dist_func = distributions_funcs_pandas(distribution)
    users_pref_dist_df = SaveAndLoad.load_user_preference_distribution(
        dataset=dataset_instance.system_name, trial=trial, fold=fold,
        distribution=distribution
    )

    users_recommendation_lists[Label.USER_ID] = users_recommendation_lists[Label.USER_ID].astype(str)
    mace_value = mace(
        users_pref_dist_df, users_recommendation_lists, items_classes_set, dist_func
    )

    results = pd.DataFrame([[
        mace_value
    ]], columns=['MACE'])

    SaveAndLoad.save_recommender_metric(
        data=results, metric="MACE",
        recommender=recommender, dataset=dataset, trial=trial, fold=fold,
        distribution=distribution, fairness=fairness, relevance=relevance,
        weight=weight, tradeoff=tradeoff, selector=selector
    )
    return "Finished"


def applying_mrmc(recommender, dataset, trial, fold, distribution, fairness, relevance, weight, tradeoff, selector):
    """
    Function that apply the evaluation metrics.
    """
    try:
        metric_file = SaveAndLoad.load_recommender_metric(
            metric="MRMC",
            recommender=recommender, dataset=dataset, trial=trial, fold=fold,
            distribution=distribution, fairness=fairness, relevance=relevance,
            weight=weight, tradeoff=tradeoff, selector=selector
        )
        # Check integrity
        if len(metric_file) == 1:
            return "AlreadyDone"
    except Exception as e:
        logger.error(" - ".join([str(e)]))

    # Load dataset
    dataset_instance = RegisteredDataset.load_dataset(dataset)
    items_set = dataset_instance.get_items()
    items_classes_set = genre_probability_approach(item_set=items_set)

    dist_func = distributions_funcs_pandas(distribution)
    fairness_func = calibration_measures_funcs(measure=fairness)

    path = PathDirFile.get_recommendation_list_file(
        dataset=dataset, recommender=recommender, trial=trial, fold=fold,
        tradeoff=tradeoff, distribution=distribution, fairness=fairness,
        relevance=relevance, tradeoff_weight=weight, select_item=selector
    )
    users_recommendation_lists = pd.read_csv(path)
    users_pref_dist_df = SaveAndLoad.load_user_preference_distribution(
        dataset=dataset_instance.system_name, trial=trial, fold=fold,
        distribution=distribution
    )

    users_recommendation_lists[Label.USER_ID] = users_recommendation_lists[Label.USER_ID].astype(str)
    mrmc_value = mrmc(
        users_pref_dist_df, users_recommendation_lists, items_classes_set, dist_func, fairness_func
    )

    results = pd.DataFrame([[
        mrmc_value
    ]], columns=['MRMC'])

    SaveAndLoad.save_recommender_metric(
        data=results, metric="MRMC",
        recommender=recommender, dataset=dataset, trial=trial, fold=fold,
        distribution=distribution, fairness=fairness, relevance=relevance,
        weight=weight, tradeoff=tradeoff, selector=selector
    )
    return "Finished"


def execution_time_fold(recommender, dataset, trial, fold,
                        distribution, fairness, relevance, weight, tradeoff, selector):
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


def applying_unexpectedness(recommender, dataset, trial, fold, distribution, fairness, relevance, weight, tradeoff, selector):
    """
    Function that apply the evaluation metrics.
    """
    exists_file = PathDirFile.get_recommender_metric_fold_file(
        recommender=recommender, dataset=dataset, trial=trial, fold=fold,
        distribution=distribution, fairness=fairness, relevance=relevance,
        tradeoff_weight=weight, tradeoff=tradeoff, select_item=selector,
        filename='UNEXPECTEDNESS.csv'
    )

    try:
        # Check integrity
        if os.path.exists(exists_file):
            metric_file = pd.read_csv(exists_file)
            if len(metric_file) == 1:
                return "AlreadyDone"
    except Exception as e:
        logger.error(" - ".join([str(e), exists_file]))

    dataset_instance = RegisteredDataset.load_dataset(dataset)

    path = PathDirFile.get_recommendation_list_file(
        dataset=dataset, recommender=recommender, trial=trial, fold=fold,
        tradeoff=tradeoff, distribution=distribution, fairness=fairness,
        relevance=relevance, tradeoff_weight=weight, select_item=selector
    )
    users_recommendation_lists = pd.read_csv(path)

    # Executing
    users_test_items = dataset_instance.get_test_transactions(trial=trial, fold=fold)
    metric_value = unexpectedness(users_recommendation_lists, users_test_items)

    results = pd.DataFrame([[
        metric_value
    ]], columns=['UNEXPECTEDNESS'])

    SaveAndLoad.save_recommender_metric(
        data=results,
        recommender=recommender, dataset=dataset, trial=trial, fold=fold,
        distribution=distribution, fairness=fairness, relevance=relevance,
        weight=weight, tradeoff=tradeoff, selector=selector,
        metric='UNEXPECTEDNESS'
    )
    return "Finished"


def applying_serendipity(recommender, dataset, trial, fold, distribution, fairness, relevance, weight, tradeoff, selector):
    """
    Function that apply the evaluation metrics.
    """
    exists_file = PathDirFile.get_recommender_metric_fold_file(
        recommender=recommender, dataset=dataset, trial=trial, fold=fold,
        distribution=distribution, fairness=fairness, relevance=relevance,
        tradeoff_weight=weight, tradeoff=tradeoff, select_item=selector,
        filename='SERENDIPITY.csv'
    )
    try:
        # Check integrity
        if os.path.exists(exists_file):
            metric_file = pd.read_csv(exists_file)
            if len(metric_file) == 1:
                return "AlreadyDone"
    except Exception as e:
        logger.error(" - ".join([str(e), exists_file]))

    dataset_instance = RegisteredDataset.load_dataset(dataset)

    path = PathDirFile.get_recommendation_list_file(
        dataset=dataset, recommender=recommender, trial=trial, fold=fold,
        tradeoff=tradeoff, distribution=distribution, fairness=fairness,
        relevance=relevance, tradeoff_weight=weight, select_item=selector
    )
    users_recommendation_lists = pd.read_csv(path)

    baseline_path = PathDirFile.get_recommendation_list_file(
        dataset=dataset, recommender=recommender, trial=trial, fold=fold,
        tradeoff=tradeoff, distribution=distribution, fairness=fairness,
        relevance=relevance, tradeoff_weight='C@0.0', select_item=selector
    )
    users_baseline_items = pd.read_csv(baseline_path)

    # Executing
    users_test_items = dataset_instance.get_test_transactions(trial=trial, fold=fold)
    metric_value = serendipity(users_recommendation_lists, users_test_items, users_baseline_items)

    results = pd.DataFrame([[
        metric_value
    ]], columns=['SERENDIPITY'])

    SaveAndLoad.save_recommender_metric(
        data=results,
        recommender=recommender, dataset=dataset, trial=trial, fold=fold,
        distribution=distribution, fairness=fairness, relevance=relevance,
        weight=weight, tradeoff=tradeoff, selector=selector,
        metric='SERENDIPITY'
    )

    return "Finished"


def applying_novelty(recommender, dataset, trial, fold, distribution, fairness, relevance, weight, tradeoff, selector):
    """
    Function that apply the evaluation metrics.
    """
    exists_file = PathDirFile.get_recommender_metric_fold_file(
        recommender=recommender, dataset=dataset, trial=trial, fold=fold,
        distribution=distribution, fairness=fairness, relevance=relevance,
        tradeoff_weight=weight, tradeoff=tradeoff, select_item=selector,
        filename='NOVELTY.csv'
    )
    try:
        # Check integrity
        if os.path.exists(exists_file):
            metric_file = pd.read_csv(exists_file)
            if len(metric_file) == 1:
                return "AlreadyDone"
    except Exception as e:
        logger.error(" - ".join([str(e), exists_file]))

    dataset_instance = RegisteredDataset.load_dataset(dataset)
    train_df = dataset_instance.get_transactions()

    path = PathDirFile.get_recommendation_list_file(
        dataset=dataset, recommender=recommender, trial=trial, fold=fold,
        tradeoff=tradeoff, distribution=distribution, fairness=fairness,
        relevance=relevance, tradeoff_weight=weight, select_item=selector
    )
    users_recommendation_lists = pd.read_csv(path)

    # Executing

    rec_set = [row['ITEM_ID'].tolist() for ix, row in users_recommendation_lists.groupby(by=['USER_ID'])]
    pop = Counter(train_df[Label.ITEM_ID].tolist())
    u = train_df[Label.USER_ID].nunique()
    metric_value, _ = recmetrics.novelty(
        predicted=rec_set, pop=pop, u=u, n=Constants.RECOMMENDATION_LIST_SIZE
    )

    results = pd.DataFrame([[
        metric_value
    ]], columns=['NOVELTY'])

    SaveAndLoad.save_recommender_metric(
        data=results,
        recommender=recommender, dataset=dataset, trial=trial, fold=fold,
        distribution=distribution, fairness=fairness, relevance=relevance,
        weight=weight, tradeoff=tradeoff, selector=selector,
        metric='NOVELTY'
    )

    return "Finished"


def applying_coverage(recommender, dataset, trial, fold, distribution, fairness, relevance, weight, tradeoff, selector):
    """
    Function that apply the evaluation metrics.
    """
    exists_file = PathDirFile.get_recommender_metric_fold_file(
        recommender=recommender, dataset=dataset, trial=trial, fold=fold,
        distribution=distribution, fairness=fairness, relevance=relevance,
        tradeoff_weight=weight, tradeoff=tradeoff, select_item=selector,
        filename='COVERAGE.csv'
    )
    try:
        # Check integrity
        if os.path.exists(exists_file):
            metric_file = pd.read_csv(exists_file)
            if len(metric_file) == 1:
                return "AlreadyDone"
    except Exception as e:
        logger.error(" - ".join([str(e), exists_file]))

    dataset_instance = RegisteredDataset.load_dataset(dataset)
    items_df = dataset_instance.get_items()

    path = PathDirFile.get_recommendation_list_file(
        dataset=dataset, recommender=recommender, trial=trial, fold=fold,
        tradeoff=tradeoff, distribution=distribution, fairness=fairness,
        relevance=relevance, tradeoff_weight=weight, select_item=selector
    )
    users_recommendation_lists = pd.read_csv(path)
    rec_set = [row['ITEM_ID'].tolist() for ix, row in users_recommendation_lists.groupby(by=['USER_ID'])]

    # Executing
    metric_value = recmetrics.prediction_coverage(
        predicted=rec_set, catalog=items_df[Label.ITEM_ID].tolist()
    )

    results = pd.DataFrame([[
        metric_value
    ]], columns=['COVERAGE'])

    SaveAndLoad.save_recommender_metric(
        data=results,
        recommender=recommender, dataset=dataset, trial=trial, fold=fold,
        distribution=distribution, fairness=fairness, relevance=relevance,
        weight=weight, tradeoff=tradeoff, selector=selector,
        metric='COVERAGE'
    )

    return "Finished"


def applying_personalization(recommender, dataset, trial, fold, distribution, fairness, relevance, weight, tradeoff, selector):
    """
    Function that apply the evaluation metrics.
    """
    exists_file = PathDirFile.get_recommender_metric_fold_file(
        recommender=recommender, dataset=dataset, trial=trial, fold=fold,
        distribution=distribution, fairness=fairness, relevance=relevance,
        tradeoff_weight=weight, tradeoff=tradeoff, select_item=selector,
        filename='PERSONALIZATION.csv'
    )
    try:
        # Check integrity
        if os.path.exists(exists_file):
            metric_file = pd.read_csv(exists_file)
            if len(metric_file) == 1:
                return "AlreadyDone"
    except Exception as e:
        logger.error(" - ".join([str(e), exists_file]))

    path = PathDirFile.get_recommendation_list_file(
        dataset=dataset, recommender=recommender, trial=trial, fold=fold,
        tradeoff=tradeoff, distribution=distribution, fairness=fairness,
        relevance=relevance, tradeoff_weight=weight, select_item=selector
    )
    users_recommendation_lists = pd.read_csv(path)
    rec_set = [row['ITEM_ID'].tolist() for ix, row in users_recommendation_lists.groupby(by=['USER_ID'])]

    # Executing
    metric_value = recmetrics.personalization(
        predicted=rec_set
    )

    results = pd.DataFrame([[
        metric_value
    ]], columns=['PERSONALIZATION'])

    SaveAndLoad.save_recommender_metric(
        data=results,
        recommender=recommender, dataset=dataset, trial=trial, fold=fold,
        distribution=distribution, fairness=fairness, relevance=relevance,
        weight=weight, tradeoff=tradeoff, selector=selector,
        metric='PERSONALIZATION'
    )

    return "Finished"
