import itertools
import logging

from searches.conformity_search import ManualConformityAlgorithmSearch
from searches.recommender_search import RecommenderSearch
from settings.labels import Label
from settings.path_dir_file import PathDirFile
from settings.save_and_load import SaveAndLoad
from utils.clocker import Clocker
from utils.input import Input
from utils.logging_settings import setup_logging
from utils.step import Step

logger = logging.getLogger(__name__)


class PierreStep2(Step):
    """
    This class is administrating the Step 2 of the framework (Hyperparameters search)
    """

    def read_the_entries(self) -> None:
        """
        This method reads the terminal entries.
        """
        self.experimental_settings = Input.step2()

    @staticmethod
    def set_the_logfile_step2(recommender: str, dataset: str) -> None:
        """
        This method is to config the log file.
        """
        # Setup Log configuration
        setup_logging(
            log_error="error.log", log_info="info.log",
            save_path=PathDirFile.set_log_search_path(
                algorithm=recommender,
                dataset=dataset
            )
        )

    def print_basic_info(self) -> None:
        """
        This method is to print basic information about the step and machine.
        """
        # Logging machine data
        logger.info("$" * 50)
        self.machine_information()
        logger.info("-" * 50)
        # Logging the experiment setup
        logger.info("-" * 50)
        logger.info("[Search Step] SEARCH FOR THE BEST PARAMETER VALUES")
        logger.info(" ".join(['>>', 'Option:', self.experimental_settings['opt']]))
        if self.experimental_settings['opt'] == Label.CONFORMITY:
            logger.info(" ".join(['>>', 'Cluster:', self.experimental_settings['cluster']]))
        elif self.experimental_settings['opt'] == Label.RECOMMENDER:
            logger.info(" ".join(['>>', 'Recommender:', str(self.experimental_settings['recommender'])]))

        logger.info(" ".join(['>>', 'Dataset:', str(self.experimental_settings['dataset'])]))
        logger.info(" ".join(['>>', 'Fold to use:', str(self.experimental_settings['fold'])]))
        logger.info(" ".join(['>>', 'Trial to use:', str(self.experimental_settings['trial'])]))

        logger.info("$" * 50)

    # ############################################################################################# #
    # ############################## Clustering Algorithm Optimization ############################ #
    # ############################################################################################# #

    def starting_cluster(self) -> None:
        """
        TODO: Docstring
        """

        # Starting the counter
        clock = Clocker()
        clock.start_count()

        # # Executing the Random Search
        search_instance = ManualConformityAlgorithmSearch(
            experimental_settings=self.experimental_settings
        )
        for algorithm in self.experimental_settings['cluster']:
            logger.info(f"Starting Algorithm: {algorithm}")
            search_instance.run(conformity_str=algorithm, recommender=self.experimental_settings['recommender'])
        #
        # Finishing the counter
        clock.finish_count()
        #
        # Saving execution time
        SaveAndLoad.save_search_conformity_time(
            data=clock.clock_data(),
            dataset=self.experimental_settings['dataset'],
            algorithm=self.experimental_settings['cluster'],
            distribution=self.experimental_settings['distribution']
        )

    # ############################################################################################# #
    # ############################# Recommender Algorithm Optimization ############################ #
    # ############################################################################################# #

    def preparing_to_batch_recommender_search(self) -> None:
        """
        TODO: Docstring
        """

        combination = [
            self.experimental_settings['recommender'], self.experimental_settings['dataset'],
            self.experimental_settings['trial'], self.experimental_settings['fold']
        ]

        starmap_params = [
            (
                recommender, dataset, trial, fold,
                self.experimental_settings['n_inter'],
                self.experimental_settings['n_jobs'],
                self.experimental_settings['n_cv']
            )
            for recommender, dataset, trial, fold in list(itertools.product(*combination))
        ]

        output = list(itertools.starmap(
            PierreStep2.starting_recommender_search,
            starmap_params
        ))

        # for recommender, dataset, trial, fold in list(itertools.product(*combination)):
        #     self.starting_recommender_search(
        #         recommender=recommender, dataset=dataset, trial=trial, fold=fold,
        #         n_inter=self.experimental_settings['n_inter'], n_jobs=self.experimental_settings['n_jobs'],
        #         n_cv=self.experimental_settings['n_cv']
        #     )

    @staticmethod
    def starting_recommender_search(
            recommender: str, dataset: str, trial: int, fold: int, n_inter: int, n_jobs: int, n_cv: int
    ) -> None:
        """
        Method to start the recommender algorithm hyperparameter search optimization.
        """

        # Starting the counter
        clock = Clocker()
        clock.start_count()

        # self.set_the_logfile_step2(dataset=dataset, recommender=recommender)

        # Executing the Random Search
        search_instance = RecommenderSearch(
            recommender=recommender, dataset=dataset, trial=trial, fold=fold,
            n_inter=n_inter, n_jobs=n_jobs, n_cv=n_cv
        )
        search_instance.fit()

        # Finishing the counter
        clock.finish_count()

        # Saving execution time
        SaveAndLoad.save_search_time(
            data=clock.clock_data(),
            dataset=dataset, algorithm=dataset
        )

    # ############################################################################################# #
    #  ################################# Main Method and Step Starts #############################  #
    # ############################################################################################# #

    def main(self) -> None:
        """
        Main method used to choice the run option.
        """
        if self.experimental_settings['opt'] == Label.CONFORMITY:
            self.starting_cluster()
        elif self.experimental_settings['opt'] == Label.RECOMMENDER:
            self.preparing_to_batch_recommender_search()
        else:
            logger.info("Option not found!")


if __name__ == '__main__':
    """
    It starts the parameter search
    """
    logger.info(" ".join(['+' * 10, 'System Starting', '+' * 10]))
    step = PierreStep2()
    step.read_the_entries()
    step.print_basic_info()
    step.main()
    logger.info(" ".join(['+' * 10, 'System shutdown', '+' * 10]))
