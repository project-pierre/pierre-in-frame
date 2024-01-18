from collections import Counter

import itertools
import logging
import os
import pandas as pd
from joblib import Parallel, delayed

from postprocessing.post_processing_step import PostProcessingStep
from settings.constants import Constants
from settings.path_dir_file import PathDirFile
from settings.save_and_load import SaveAndLoad
from utils.input import Input
from utils.logging_settings import setup_logging
from utils.step import Step

logger = logging.getLogger(__name__)


class PierreStep4(Step):
    """
    This class is administrating the Step 4 of the framework (Post-Processing)
    """

    def read_the_entries(self) -> None:
        """
        This method reads the terminal entries.
        """
        self.experimental_settings = Input.step4()

    @staticmethod
    def set_the_logfile_by_instance(
        dataset: str, algorithm: str, trial: int, fold: int,
        tradeoff: str, distribution: str, calibration: str, relevance: str,
        weight: str, selector: str
    ) -> None:
        """
        This method is to config the log file.
        """
        # Setup Log configuration
        setup_logging(
            log_error="error.log", log_info="info.log",
            save_path=PathDirFile.set_log_postprocessing_path(
                recommender=algorithm, trial=trial, fold=fold, dataset=dataset,
                tradeoff=tradeoff, distribution=distribution, fairness=calibration,
                relevance=relevance, tradeoff_weight=weight, select_item=selector
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
        logger.info("[POST-PROCESSING STEP] - Creating recommendation list")
        logger.info(" ".join(['>>', 'Dataset:', dataset]))
        logger.info(" ".join(['>>', 'Trial:', str(trial)]))
        logger.info(" ".join(['>>', 'Fold:', str(fold)]))
        logger.info(" ".join(['>>', 'Algorithm:', algorithm]))
        logger.info("$" * 50)
        logger.info("$" * 50)

    def checkpoint_verification(
        self, recommender, fold, trial, dataset,
        tradeoff, distribution, calibration, relevance, weight, selector
    ) -> bool:

        path = PathDirFile.get_recommendation_list_file(
            dataset=dataset, recommender=recommender, trial=trial, fold=fold,
            tradeoff=tradeoff, distribution=distribution, fairness=calibration,
            relevance=relevance, tradeoff_weight=weight, select_item=selector
        )

        # Check integrity.
        if os.path.exists(path):
            try:
                users_recommendation_lists = pd.read_csv(path)
                if len(users_recommendation_lists) > 100:
                    return True
                else:
                    return False
            except Exception as e:
                logger.error(" - ".join([str(e), path]))

    def starting_postprocessing(self, recommender, fold, trial, dataset, tradeoff, distribution, calibration, relevance,
                                weight, selector, list_size, alpha, d) -> str:
        """
        TODO: Docstring
        """
        self.set_the_logfile_by_instance(
            dataset=dataset, trial=trial, fold=fold, algorithm=recommender,
            tradeoff=tradeoff, distribution=distribution, calibration=calibration,
            relevance=relevance, weight=weight, selector=selector
        )
        # self.print_basic_info_by_instance(
        #     dataset=dataset, trial=trial, fold=fold, algorithm=recommender
        # )

        if self.experimental_settings["checkpoint"] == "YES" and self.checkpoint_verification(
            dataset=dataset, trial=trial, fold=fold, recommender=recommender,
            tradeoff=tradeoff, distribution=distribution, calibration=calibration,
            relevance=relevance, weight=weight, selector=selector
        ):
            logger.info(">> Already Done... " + "-".join(
                [dataset,
                 'trial-' + str(trial), 'fold-' + str(fold), recommender,
                 tradeoff, distribution, relevance, selector,
                 calibration, tradeoff])
            )
            return "Already Done"

        # Starting the counter
        self.start_count()

        # Instancing the post-processing
        pierre = PostProcessingStep(
            recommender=recommender, dataset_name=dataset, trial=trial, fold=fold,
            tradeoff_component=tradeoff, distribution_component=distribution,
            fairness_component=calibration, relevance_component=relevance,
            tradeoff_weight_component=weight, selector_component=selector,
            list_size=list_size, alpha=alpha, d=d
        )
        pierre.run()

        # Finishing the counter
        self.finish_count()

        # Saving execution time
        SaveAndLoad.save_postprocessing_time(
            data=self.clock_data(),
            dataset=dataset, trial=trial, fold=fold, recommender=recommender,
            tradeoff=tradeoff, distribution=distribution, fairness=calibration,
            relevance=relevance, tradeoff_weight=weight, select_item=selector
        )

        # Finishing the step
        logger.info(" ".join(['->>', 'Time Execution:', str(self.get_total_time_formatted())]))
        return "Finished"

    def main(self) -> None:
        combination = [
            self.experimental_settings['recommender'], self.experimental_settings['dataset'],
            self.experimental_settings['fold'], self.experimental_settings['trial'],
            self.experimental_settings['tradeoff'], self.experimental_settings['relevance'],
            self.experimental_settings['distribution'], self.experimental_settings['selector'],
            self.experimental_settings['weight'], self.experimental_settings['fairness'],
            self.experimental_settings['list_size'], self.experimental_settings['alpha'],
            self.experimental_settings['d']
        ]

        load = Parallel(n_jobs=Constants.N_CORES)(
            delayed(self.starting_postprocessing)(
                recommender=recommender, dataset=dataset, trial=trial, fold=fold,
                tradeoff=tradeoff, relevance=relevance, distribution=distribution,
                selector=selector, weight=weight, calibration=calibration,
                list_size=list_size, alpha=alpha, d=d
            ) for
            recommender, dataset, fold, trial, tradeoff, relevance, distribution, selector, weight, calibration, list_size, alpha, d
            in list(itertools.product(*combination))
        )

        jobs = dict(Counter(load))
        logger.info(jobs)


if __name__ == '__main__':
    """
    It starts the post-processing step
    """
    logger.info(" ".join(['+' * 10, 'System Starting', '+' * 10]))
    step = PierreStep4()
    step.read_the_entries()
    step.main()
    logger.info(" ".join(['+' * 10, 'System shutdown', '+' * 10]))
