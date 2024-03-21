from collections import Counter

import logging
import pandas as pd
import recmetrics

from checkpoint_verification import CheckpointVerification
from datasets.registred_datasets import RegisteredDataset
from scikit_pierre.metrics.evaluation import (mace, mean_average_precision,
                                              rank_miscalibration, miscalibration,
                                              mean_reciprocal_rank, serendipity, unexpectedness,
                                              Miscalibration, MeanAbsoluteCalibrationError
                                              )
from settings.constants import Constants
from settings.labels import Label
from settings.path_dir_file import PathDirFile
from settings.save_and_load import SaveAndLoad

logger = logging.getLogger(__name__)


def applying_mrr(
    recommender: str, dataset: str, trial: int, fold: int,
    distribution: str, fairness: str, relevance: str, weight: str, tradeoff: str, selector: str,
    checkpoint: str
):
    """
    Function that apply the evaluation metrics.
    """

    system_name = "-".join([
        dataset, 'trial-' + str(trial), 'fold-' + str(fold), recommender,
        tradeoff, distribution, relevance, selector, fairness, tradeoff, "-->", Label.MRR
    ])

    if checkpoint == "YES" and CheckpointVerification.unit_step5_recommendation_verification(
            dataset=dataset, trial=trial, fold=fold,
            metric=Label.MRR, recommender=recommender,
            distribution=distribution, fairness=fairness, relevance=relevance,
            weight=weight, tradeoff=tradeoff, selector=selector
    ):
        logger.info(">> Already Done... " + system_name)
        return "Already Done"

    dataset_instance = RegisteredDataset.load_dataset(dataset)

    path = PathDirFile.get_recommendation_list_file(
        dataset=dataset, recommender=recommender, trial=trial, fold=fold,
        tradeoff=tradeoff, distribution=distribution, fairness=fairness,
        relevance=relevance, tradeoff_weight=weight, select_item=selector
    )
    users_recommendation_lists = pd.read_csv(path)

    # Executing
    users_test_items = dataset_instance.get_test_transactions(trial=trial, fold=fold)
    mrr_value = mean_reciprocal_rank(users_recommendation_lists, users_test_items)

    results = pd.DataFrame([[
        mrr_value
    ]], columns=[Label.MRR])

    SaveAndLoad.save_recommender_metric(
        data=results, metric=Label.MRR,
        recommender=recommender, dataset=dataset, trial=trial, fold=fold,
        distribution=distribution, fairness=fairness, relevance=relevance,
        weight=weight, tradeoff=tradeoff, selector=selector
    )
    return "Finished"


def applying_map(
    recommender: str, dataset: str, trial: int, fold: int,
    distribution: str, fairness: str, relevance: str, weight: str, tradeoff: str, selector: str,
    checkpoint: str
):
    """
    Function that apply the evaluation metrics.
    """

    system_name = "-".join([
        dataset, 'trial-' + str(trial), 'fold-' + str(fold), recommender,
        tradeoff, distribution, relevance, selector, fairness, tradeoff, "-->", Label.MAP
    ])

    if checkpoint == "YES" and CheckpointVerification.unit_step5_recommendation_verification(
            dataset=dataset, trial=trial, fold=fold,
            metric=Label.MAP, recommender=recommender,
            distribution=distribution, fairness=fairness, relevance=relevance,
            weight=weight, tradeoff=tradeoff, selector=selector
    ):
        logger.info(">> Already Done... " + system_name)
        return "Already Done"

    dataset_instance = RegisteredDataset.load_dataset(dataset)

    path = PathDirFile.get_recommendation_list_file(
        dataset=dataset, recommender=recommender, trial=trial, fold=fold,
        tradeoff=tradeoff, distribution=distribution, fairness=fairness,
        relevance=relevance, tradeoff_weight=weight, select_item=selector
    )
    users_recommendation_lists = pd.read_csv(path)

    # Executing
    users_test_items = dataset_instance.get_test_transactions(trial=trial, fold=fold)
    map_value = mean_average_precision(users_recommendation_lists, users_test_items)

    results = pd.DataFrame([[
        map_value
    ]], columns=[Label.MAP])

    SaveAndLoad.save_recommender_metric(
        data=results, metric=Label.MAP,
        recommender=recommender, dataset=dataset, trial=trial, fold=fold,
        distribution=distribution, fairness=fairness, relevance=relevance,
        weight=weight, tradeoff=tradeoff, selector=selector
    )
    return "Finished"


def applying_mace(
    recommender: str, dataset: str, trial: int, fold: int,
    distribution: str, fairness: str, relevance: str, weight: str, tradeoff: str, selector: str,
    checkpoint: str
):
    """
    Function that apply the evaluation metrics.
    """

    system_name = "-".join([
        dataset, 'trial-' + str(trial), 'fold-' + str(fold), recommender,
        tradeoff, distribution, relevance, selector, fairness, tradeoff, "-->", Label.MACE
    ])

    if checkpoint == "YES" and CheckpointVerification.unit_step5_recommendation_verification(
            dataset=dataset, trial=trial, fold=fold,
            metric=Label.MACE, recommender=recommender,
            distribution=distribution, fairness=fairness, relevance=relevance,
            weight=weight, tradeoff=tradeoff, selector=selector
    ):
        logger.info(">> Already Done... " + system_name)
        return "Already Done"

    dataset_instance = RegisteredDataset.load_dataset(dataset)

    path = PathDirFile.get_recommendation_list_file(
        dataset=dataset, recommender=recommender, trial=trial, fold=fold,
        tradeoff=tradeoff, distribution=distribution, fairness=fairness,
        relevance=relevance, tradeoff_weight=weight, select_item=selector
    )
    users_recommendation_lists = pd.read_csv(path)
    users_recommendation_lists[Label.USER_ID] = users_recommendation_lists[Label.USER_ID].astype(str)

    # Executing
    items_set = dataset_instance.get_items()

    # Get the users' preferences set
    users_preference_set = dataset_instance.get_train_transactions(
        trial=trial, fold=fold
    )

    set_1 = set([str(ix) for ix in users_recommendation_lists['USER_ID'].unique().tolist()])
    set_2 = set([str(ix) for ix in users_preference_set['USER_ID'].unique().tolist()])

    if set_1 != set_2:
        print(set_1 - set_2)
        print("Size rec list: ", str(len(set_1)))
        print("Size profiles: ", str(len(set_2)))
        msg = "".join(["Size rec list: ", str(len(set_1)),
                       "----", "Size profiles: ", str(len(set_2)),
                       "---------------------", set_1 - set_2])
        raise Exception(msg)

    # mace_value = mace(
    #     users_preference_set=users_preference_set,
    #     users_recommendation_lists=users_recommendation_lists,
    #     items_set_df=items_set, distribution=distribution
    # )

    instance = MeanAbsoluteCalibrationError(
        users_preference_set=users_preference_set,
        users_recommendation_lists=users_recommendation_lists,
        items_set_df=items_set, distribution=distribution
    )
    _value = instance.main()
    results = pd.DataFrame([[
        _value
    ]], columns=[Label.MACE])

    SaveAndLoad.save_recommender_metric(
        data=results, metric=Label.MACE,
        recommender=recommender, dataset=dataset, trial=trial, fold=fold,
        distribution=distribution, fairness=fairness, relevance=relevance,
        weight=weight, tradeoff=tradeoff, selector=selector
    )
    return "Finished"


def applying_MC(
    recommender: str, dataset: str, trial: int, fold: int,
    distribution: str, fairness: str, relevance: str, weight: str, tradeoff: str, selector: str,
    checkpoint: str
):
    """
    Function that apply the evaluation metrics.
    """

    system_name = "-".join([
        dataset, 'trial-' + str(trial), 'fold-' + str(fold), recommender,
        tradeoff, distribution, relevance, selector, fairness, tradeoff, "-->", Label.MC
    ])

    if checkpoint == "YES" and CheckpointVerification.unit_step5_recommendation_verification(
            dataset=dataset, trial=trial, fold=fold,
            metric=Label.MC, recommender=recommender,
            distribution=distribution, fairness=fairness, relevance=relevance,
            weight=weight, tradeoff=tradeoff, selector=selector
    ):
        logger.info(">> Already Done... " + system_name)
        return "Already Done"

    dataset_instance = RegisteredDataset.load_dataset(dataset)

    path = PathDirFile.get_recommendation_list_file(
        dataset=dataset, recommender=recommender, trial=trial, fold=fold,
        tradeoff=tradeoff, distribution=distribution, fairness=fairness,
        relevance=relevance, tradeoff_weight=weight, select_item=selector
    )
    users_recommendation_lists = pd.read_csv(path)
    users_recommendation_lists[Label.USER_ID] = users_recommendation_lists[Label.USER_ID].astype(str)

    # Executing
    items_set = dataset_instance.get_items()

    # Get the users' preferences set
    users_preference_set = dataset_instance.get_train_transactions(
        trial=trial, fold=fold
    )

    set_1 = set([str(ix) for ix in users_recommendation_lists['USER_ID'].unique().tolist()])
    set_2 = set([str(ix) for ix in users_preference_set['USER_ID'].unique().tolist()])

    if set_1 != set_2:
        print(set_1 - set_2)
        print("Size rec list: ", str(len(set_1)))
        print("Size profiles: ", str(len(set_2)))
        msg = "".join(["Size rec list: ", str(len(set_1)),
                       "----", "Size profiles: ", str(len(set_2)),
                       "---------------------", set_1 - set_2])
        raise Exception(msg)

    _value = miscalibration(
        users_preference_set=users_preference_set,
        users_recommendation_lists=users_recommendation_lists,
        items_set_df=items_set, distribution=distribution, distance_func_name=fairness
    )

    instance = Miscalibration(
        users_preference_set=users_preference_set,
        users_recommendation_lists=users_recommendation_lists,
        items_set_df=items_set, distribution=distribution, distance_func_name=fairness
    )
    _value = instance.main()
    results = pd.DataFrame([[
        _value
    ]], columns=[Label.MC])

    SaveAndLoad.save_recommender_metric(
        data=results, metric=Label.MC,
        recommender=recommender, dataset=dataset, trial=trial, fold=fold,
        distribution=distribution, fairness=fairness, relevance=relevance,
        weight=weight, tradeoff=tradeoff, selector=selector
    )
    return "Finished"


def applying_rmc(
    recommender: str, dataset: str, trial: int, fold: int,
    distribution: str, fairness: str, relevance: str, weight: str, tradeoff: str, selector: str,
    checkpoint: str
):
    """
    Function that apply the evaluation metrics.
    """

    system_name = "-".join([
        dataset, 'trial-' + str(trial), 'fold-' + str(fold), recommender,
        tradeoff, distribution, relevance, selector, fairness, tradeoff, "-->", Label.RMC
    ])

    if checkpoint == "YES" and CheckpointVerification.unit_step5_recommendation_verification(
            dataset=dataset, trial=trial, fold=fold,
            metric=Label.RMC, recommender=recommender,
            distribution=distribution, fairness=fairness, relevance=relevance,
            weight=weight, tradeoff=tradeoff, selector=selector
    ):
        logger.info(">> Already Done... " + system_name)
        return "Already Done"

    dataset_instance = RegisteredDataset.load_dataset(dataset)

    path = PathDirFile.get_recommendation_list_file(
        dataset=dataset, recommender=recommender, trial=trial, fold=fold,
        tradeoff=tradeoff, distribution=distribution, fairness=fairness,
        relevance=relevance, tradeoff_weight=weight, select_item=selector
    )
    users_recommendation_lists = pd.read_csv(path)
    users_recommendation_lists[Label.USER_ID] = users_recommendation_lists[Label.USER_ID].astype(str)

    # Executing
    items_set = dataset_instance.get_items()

    # Get the users' preferences set
    users_preference_set = dataset_instance.get_train_transactions(
        trial=trial, fold=fold
    )

    set_1 = set([str(ix) for ix in users_recommendation_lists['USER_ID'].unique().tolist()])
    set_2 = set([str(ix) for ix in users_preference_set['USER_ID'].unique().tolist()])

    if set_1 != set_2:
        print(set_1 - set_2)
        print("Size rec list: ", str(len(set_1)))
        print("Size profiles: ", str(len(set_2)))
        msg = "".join(["Size rec list: ", str(len(set_1)),
                       "----", "Size profiles: ", str(len(set_2)),
                       "---------------------", set_1 - set_2])
        raise Exception(msg)

    mace_value = rank_miscalibration(
        users_preference_set=users_preference_set, users_recommendation_lists=users_recommendation_lists,
        items_set_df=items_set, distribution=distribution, distance_func_name=fairness
    )
    results = pd.DataFrame([[
        mace_value
    ]], columns=[Label.RMC])

    SaveAndLoad.save_recommender_metric(
        data=results, metric=Label.RMC,
        recommender=recommender, dataset=dataset, trial=trial, fold=fold,
        distribution=distribution, fairness=fairness, relevance=relevance,
        weight=weight, tradeoff=tradeoff, selector=selector
    )
    return "Finished"


def applying_mrmc(
    recommender: str, dataset: str, trial: int, fold: int,
    distribution: str, fairness: str, relevance: str, weight: str, tradeoff: str, selector: str,
    checkpoint: str
):
    """
    Function that apply the evaluation metrics.
    """

    system_name = "-".join([
        dataset, 'trial-' + str(trial), 'fold-' + str(fold), recommender,
        tradeoff, distribution, relevance, selector, fairness, tradeoff, "-->", Label.MRMC
    ])

    if checkpoint == "YES" and CheckpointVerification.unit_step5_recommendation_verification(
            dataset=dataset, trial=trial, fold=fold,
            metric=Label.MRMC, recommender=recommender,
            distribution=distribution, fairness=fairness, relevance=relevance,
            weight=weight, tradeoff=tradeoff, selector=selector
    ):
        logger.info(">> Already Done... " + system_name)
        return "Already Done"

    # Load dataset
    # dataset_instance = RegisteredDataset.load_dataset(dataset)
    # items_set = dataset_instance.get_items()
    # items_classes_set = genre_probability_approach(item_set=items_set)
    #
    # dist_func = distributions_funcs_pandas(distribution)
    # fairness_func = calibration_measures_funcs(measure=fairness)
    #
    # path = PathDirFile.get_recommendation_list_file(
    #     dataset=dataset, recommender=recommender, trial=trial, fold=fold,
    #     tradeoff=tradeoff, distribution=distribution, fairness=fairness,
    #     relevance=relevance, tradeoff_weight=weight, select_item=selector
    # )
    # users_recommendation_lists = pd.read_csv(path)
    # users_pref_dist_df = SaveAndLoad.load_user_preference_distribution(
    #     dataset=dataset_instance.system_name, trial=trial, fold=fold,
    #     distribution=distribution
    # )
    #
    # users_recommendation_lists[Label.USER_ID] = users_recommendation_lists[Label.USER_ID].astype(str)
    # mrmc_value = mrmc(
    #     users_pref_dist_df, users_recommendation_lists, items_classes_set, dist_func, fairness_func
    # )
    #
    # results = pd.DataFrame([[
    #     mrmc_value
    # ]], columns=[Label.MRMC])
    #
    # SaveAndLoad.save_recommender_metric(
    #     data=results, metric=Label.MRMC,
    #     recommender=recommender, dataset=dataset, trial=trial, fold=fold,
    #     distribution=distribution, fairness=fairness, relevance=relevance,
    #     weight=weight, tradeoff=tradeoff, selector=selector
    # )
    return "Finished"


def applying_unexpectedness(
    recommender: str, dataset: str, trial: int, fold: int,
    distribution: str, fairness: str, relevance: str, weight: str, tradeoff: str, selector: str,
    checkpoint: str
):
    """
    Function that apply the evaluation metrics.
    """

    system_name = "-".join([
        dataset, 'trial-' + str(trial), 'fold-' + str(fold), recommender,
        tradeoff, distribution, relevance, selector, fairness, tradeoff, "-->", Label.UNEXPECTEDNESS
    ])

    if checkpoint == "YES" and CheckpointVerification.unit_step5_recommendation_verification(
            dataset=dataset, trial=trial, fold=fold,
            metric=Label.UNEXPECTEDNESS, recommender=recommender,
            distribution=distribution, fairness=fairness, relevance=relevance,
            weight=weight, tradeoff=tradeoff, selector=selector
    ):
        logger.info(">> Already Done... " + system_name)
        return "Already Done"

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
    ]], columns=[Label.UNEXPECTEDNESS])

    SaveAndLoad.save_recommender_metric(
        data=results,
        recommender=recommender, dataset=dataset, trial=trial, fold=fold,
        distribution=distribution, fairness=fairness, relevance=relevance,
        weight=weight, tradeoff=tradeoff, selector=selector,
        metric=Label.UNEXPECTEDNESS
    )
    return "Finished"


def applying_serendipity(
    recommender: str, dataset: str, trial: int, fold: int,
    distribution: str, fairness: str, relevance: str, weight: str, tradeoff: str, selector: str,
    checkpoint: str
):
    """
    Function that apply the evaluation metrics.
    """

    system_name = "-".join([
        dataset, 'trial-' + str(trial), 'fold-' + str(fold), recommender,
        tradeoff, distribution, relevance, selector, fairness, tradeoff, "-->", Label.SERENDIPITY
    ])

    if checkpoint == "YES" and CheckpointVerification.unit_step5_recommendation_verification(
            dataset=dataset, trial=trial, fold=fold,
            metric=Label.SERENDIPITY, recommender=recommender,
            distribution=distribution, fairness=fairness, relevance=relevance,
            weight=weight, tradeoff=tradeoff, selector=selector
    ):
        logger.info(">> Already Done... " + system_name)
        return "Already Done"

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
    ]], columns=[Label.SERENDIPITY])

    SaveAndLoad.save_recommender_metric(
        data=results,
        recommender=recommender, dataset=dataset, trial=trial, fold=fold,
        distribution=distribution, fairness=fairness, relevance=relevance,
        weight=weight, tradeoff=tradeoff, selector=selector,
        metric=Label.SERENDIPITY
    )

    return "Finished"


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

    rec_set = [row[Label.ITEM_ID].tolist() for ix, row in users_recommendation_lists.groupby(by=[Label.USER_ID])]
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
    rec_set = [row[Label.ITEM_ID].tolist() for ix, row in users_recommendation_lists.groupby(by=[Label.USER_ID])]

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
        tradeoff, distribution, relevance, selector, fairness, tradeoff, "-->", Label.PERSONALIZATION
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
    rec_set = [row[Label.ITEM_ID].tolist() for ix, row in users_recommendation_lists.groupby(by=[Label.USER_ID])]

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
