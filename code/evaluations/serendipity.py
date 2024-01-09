from datasets.registred_datasets import RegisteredDataset
from scikit_pierre.metrics.serendipity import serendipity
from settings.constants import Constants
from settings.path_dir_file import PathDirFile
import pandas as pd


def serendipity_interface(recommender, dataset,
                          distribution, fairness, relevance, weight, tradeoff, selector):
    # Load dataset
    dataset_instance = RegisteredDataset.load_dataset(dataset)
    metric_values = []
    for trial in list(range(1, Constants.N_TRIAL_VALUE + 1)):
        for fold in list(range(1, Constants.K_FOLDS_VALUE + 1)):
            path = PathDirFile.get_recommendation_list_file(
                dataset=dataset, recommender=recommender, trial=trial, fold=fold,
                tradeoff=tradeoff, distribution=distribution, fairness=fairness,
                relevance=relevance, tradeoff_weight=weight, select_item=selector
            )
            baseline_path = PathDirFile.get_recommendation_list_file(
                dataset=dataset, recommender=recommender, trial=trial, fold=fold,
                tradeoff='C@0.0', distribution=distribution, fairness=fairness,
                relevance=relevance, tradeoff_weight=weight, select_item=selector
            )
            users_recommendation_list = pd.read_csv(path)
            users_baseline_items = pd.read_csv(baseline_path)
            users_test_items = dataset_instance.get_test_transactions(trial=trial, fold=fold)
            metric_values.append(serendipity(users_recommendation_list, users_test_items, users_baseline_items))
    return sum(metric_values)/len(metric_values)
