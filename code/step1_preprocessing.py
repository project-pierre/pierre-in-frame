import itertools
import logging

import pandas as pd
from joblib import Parallel, delayed

from datasets.registred_datasets import RegisteredDataset
from graphics.dataset_chart import DatasetChart
from scikit_pierre.classes.genre import genre_probability_approach
from scikit_pierre.distributions.accessible import distributions_funcs_pandas
from settings.constants import Constants
from settings.labels import Label
from settings.path_dir_file import PathDirFile
from settings.save_and_load import SaveAndLoad
from utils.input import Input
from utils.logging_settings import setup_logging
from utils.step import Step

logger = logging.getLogger(__name__)


class PierreStep1(Step):
    """
    This class is for administrating Step 1 of the framework.
    """

    def read_the_entries(self):
        """
        This method reads the terminal entries.
        """
        self.experimental_settings = Input.step1()

    def set_the_logfile(self):
        """
        This method is to config the log file.
        """
        # Setup Log configuration
        setup_logging(
            save_path=PathDirFile.log_preprocessing_path(dataset=self.experimental_settings['dataset'])
        )

    def print_basic_info(self):
        """
        This method is to print basic information about the step and machine.
        """
        # Logging machine data
        logger.info("$" * 50)
        self.machine_information()
        logger.info("-" * 50)
        # Logging the experiment setup
        logger.info(f"> DATASET (PREPROCESSING STEP) - {self.experimental_settings['opt']}")
        logger.info(" ".join(['>>', 'Dataset:', self.experimental_settings['dataset']]))
        logger.info(" ".join(['>>', 'Number of Folds:', str(self.experimental_settings['n_folds'])]))
        logger.info(" ".join(['>>', 'Number of Trials:', str(self.experimental_settings['n_trials'])]))
        logger.info("$" * 50)

    def create_folds(self):
        """
        This method is to split the data and create the folds.
        """
        # Starting the counter
        self.start_count()
        # Executing the pre-processing
        RegisteredDataset.preprocessing(
            dataset=self.experimental_settings['dataset'],
            n_trials=self.experimental_settings['n_trials'],
            n_folds=self.experimental_settings['n_folds']
        )

        # Finishing the counter
        self.finish_count()

        # Saving execution time
        SaveAndLoad.save_preprocessing_time(
            data=self.clock_data(),
            dataset=self.experimental_settings['dataset']
        )

        # Finishing the step
        logger.info(" ".join(['->>', 'Time Execution:', str(self.get_total_time())]))

    def create_charts(self):
        """
        This method has the function of generating graphics with the dataset analyzes.
        """
        # Loading dataset information
        dt_chat = DatasetChart(self.experimental_settings['dataset'])
        dt_chat.item_long_tail()
        dt_chat.genres()

    def create_analyzes(self):
        """
        This method is to lead with the dataset numbers.
        """

        # Load the dataset
        dataset_instance = RegisteredDataset.load_dataset(
            dataset=self.experimental_settings['dataset']
        )
        # Print the Raw dataset information
        dataset_instance.raw_data_basic_info()
        # Print the Clean dataset information
        dataset_instance.clean_data_basic_info()

    def create_distribution(self):
        """
        This method is to lead with the distribution file.
        """

        # It creates the combination of process entries
        combination = [
            [self.experimental_settings['dataset']], self.experimental_settings['trial'],
            self.experimental_settings['fold'], [self.experimental_settings['distribution']]
        ]

        # Start the processes in parallel using joblib
        load = Parallel(n_jobs=Constants.N_CORES)(
            delayed(self.compute_distribution)(
                dataset=dataset, trial=trial, fold=fold, distribution=distribution
            ) for dataset, trial, fold, distribution
            in list(itertools.product(*combination)))

    @staticmethod
    def compute_distribution(dataset, trial, fold, distribution):
        """
        This method is to compute the preference distribution.
        """

        # Load the dataset
        dataset_instance = RegisteredDataset.load_dataset(dataset)

        # Get the items classes
        items_classes_set = genre_probability_approach(item_set=dataset_instance.get_items())

        # Get the users' preferences set
        users_preference_set = dataset_instance.get_train_transactions(
            trial=trial, fold=fold
        )

        # Set the used distribution
        dist_func = distributions_funcs_pandas(distribution=distribution)

        # Group the preferences by user
        grouped_users_preference_set = users_preference_set.groupby(by=[Label.USER_ID])

        # Compute the distribution to all users
        users_pref_dist_df = pd.concat([
            dist_func(
                user_id=user_id,
                user_pref_set=user_pref_set,
                item_classes_set=items_classes_set
            ) for user_id, user_pref_set in grouped_users_preference_set
        ])

        # Save the distributions
        SaveAndLoad.save_user_preference_distribution(
            data=users_pref_dist_df, dataset=dataset, fold=fold, trial=trial, distribution=distribution
        )

        logger.info(" ... ".join([
            '->> ', 'Compute Distribution Finished to: ', dataset, distribution, str(trial), str(fold)
        ]))

    def main(self):
        """
        Main method used to choice the run option.
        """
        if self.experimental_settings['opt'] == Label.DATASET_CHART:
            self.create_charts()
        elif self.experimental_settings['opt'] == Label.DATASET_ANALYZE:
            self.create_analyzes()
        elif self.experimental_settings['opt'] == Label.DATASET_DISTRIBUTION:
            self.create_distribution()
        elif self.experimental_settings['opt'] == Label.DATASET_SPLIT:
            self.create_folds()
        else:
            print("Option not found!")


if __name__ == '__main__':
    """
    It starts the pre-processing step
    """
    logger.info(" ".join(['+' * 10, 'System Starting', '+' * 10]))
    step = PierreStep1()
    step.read_the_entries()
    step.set_the_logfile()
    step.print_basic_info()
    step.main()
    logger.info(" ".join(['+' * 10, 'System Shutdown', '+' * 10]))
