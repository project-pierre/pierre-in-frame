import itertools

import logging

from joblib import Parallel, delayed

from processing.implicit_recommender_algorithms import ImplicitRecommenderAlgorithm
from processing.surprise_recommender_algorithms import SurpriseRecommenderAlgorithm
from settings.constants import Constants
from settings.labels import Label
from utils.logging_settings import setup_logging
from settings.path_dir_file import PathDirFile
from settings.save_and_load import SaveAndLoad
from utils.input import Input
from utils.step import Step

logger = logging.getLogger(__name__)


class PierreStep3(Step):
    """
    This class is administrating the Step 3 of the framework (Processing)
    """

    def read_the_entries(self) -> None:
        """
        This method reads the terminal entries.
        """
        self.experimental_settings = Input.step3()

    @staticmethod
    def set_the_logfile_by_instance(dataset: str, algorithm: str, trial: int, fold: int) -> None:
        """
        This method is to config the log file.
        """
        # Setup Log configuration
        setup_logging(
            log_error="error.log", log_info="info.log",
            save_path=PathDirFile.set_log_processing_path(
                algorithm=algorithm,
                trial=trial,
                fold=fold,
                dataset=dataset
            )
        )

    def print_basic_info_by_instance(self, dataset: str, algorithm: str, trial: int, fold: int) -> None:
        """
        This method is to print basic information about the step and machine.
        """

        logger.info("$" * 50)
        logger.info("$" * 50)
        # Logging machine data
        self.machine_information()
        logger.info("-" * 50)

        # Logging the experiment setup
        logger.info("[PROCESSING STEP] - RECOMMENDER ALGORITHM")
        logger.info(" ".join(['>>', 'Dataset:', dataset]))
        logger.info(" ".join(['>>', 'Trial:', str(trial)]))
        logger.info(" ".join(['>>', 'Fold:', str(fold)]))
        logger.info(" ".join(['>>', 'Algorithm:', algorithm]))
        logger.info("$" * 50)
        logger.info("$" * 50)

    def main(self) -> None:
        """
        Main method used to choice the run option.
        """
        self.recommender_parallelization()

    def recommender_parallelization(self) -> None:
        """
        Main method to start the processing step in parallel.
        """
        combination = [
            self.experimental_settings['recommender'], self.experimental_settings['dataset'],
            self.experimental_settings['fold'], self.experimental_settings['trial']
        ]

        # Starting the recommender algorithm
        Parallel(n_jobs=self.experimental_settings['n_jobs'])(
            delayed(self.starting_recommender)(
                recommender=recommender, dataset=dataset, trial=trial, fold=fold
            ) for recommender, dataset, fold, trial in list(itertools.product(*combination))
        )

        # for recommender, dataset, fold, trial in list(itertools.product(*combination)):
        #     self.starting_recommender(recommender=recommender, dataset=dataset, trial=trial, fold=fold)

        # Finishing the Step
        logger.info(" ".join(['+' * 10, 'System shutdown', '+' * 10]))

    def starting_recommender(self, dataset: str, recommender: str, trial: int, fold: int) -> None:
        """
        Function to starting the recommender algorithm.

        :param dataset: A string that's representing the dataset name.
        :param recommender: A string that's representing the recommender algorithm name.
        :param trial: The trial number.
        :param fold: The fold number.
        """
        self.set_the_logfile_by_instance(
            dataset=dataset, trial=trial, fold=fold, algorithm=recommender
        )
        self.print_basic_info_by_instance(
            dataset=dataset, trial=trial, fold=fold, algorithm=recommender
        )

        # Starting the counter
        self.start_count()

        if recommender in Label.SURPRISE_RECOMMENDERS:
            # Executing the processing step
            recommender_algorithm = SurpriseRecommenderAlgorithm(
                dataset_name=dataset, trial=trial, fold=fold, recommender_name=recommender,
                metric=self.experimental_settings['metric'],
                list_size=self.experimental_settings['list_size']
            )
            recommender_algorithm.run()
        elif recommender in Label.IMPLICIT_RECOMMENDERS:
            recommender_algorithm = ImplicitRecommenderAlgorithm(
                dataset_name=dataset, trial=trial, fold=fold, recommender_name=recommender,
                list_size=self.experimental_settings['list_size']
            )
            recommender_algorithm.run()
        else:
            pass

        # Finishing the counter
        self.finish_count()

        # Saving execution time
        SaveAndLoad.save_processing_time(
            data=self.clock_data(),
            dataset=dataset, trial=trial, fold=fold, algorithm=recommender
        )
        # Finishing the step
        logger.info(" ".join(['->>', 'Time Execution:', str(self.get_total_time())]))


if __name__ == '__main__':
    """
    It starts the processing step
    """
    logger.info(" ".join(['+' * 10, 'System Starting', '+' * 10]))
    step = PierreStep3()
    step.read_the_entries()
    step.main()
    logger.info(" ".join(['+' * 10, 'System shutdown', '+' * 10]))
