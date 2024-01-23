import multiprocessing
from collections import Counter

import itertools
import logging
from joblib import Parallel, delayed

from checkpoint_verification import CheckpointVerification
from postprocessing.post_processing_step import PostProcessingStep
from settings.path_dir_file import PathDirFile
from settings.save_and_load import SaveAndLoad
from utils.clocker import Clocker
from utils.input import Input
from utils.logging_settings import setup_logging
from utils.step import Step

logger = logging.getLogger(__name__)


def starting_postprocessing(
        recommender: str, fold: int, trial: int, dataset: str,
        tradeoff: str, distribution: str, calibration: str, relevance: str,
        weight: str, selector: str, list_size: int, alpha: int, d: int, checkpoint: str
) -> str:
    """
    TODO: Docstring
    """
    # PierreStep4.set_the_logfile_by_instance(
    #     dataset=dataset, trial=trial, fold=fold, algorithm=recommender,
    #     tradeoff=tradeoff, distribution=distribution, calibration=calibration,
    #     relevance=relevance, weight=weight, selector=selector
    # )
    system_name = "-".join([
        dataset, 'trial-' + str(trial), 'fold-' + str(fold), recommender,
        tradeoff, distribution, relevance, selector, calibration, tradeoff
    ])

    if checkpoint == "YES" and CheckpointVerification.unit_step4_verification(
            dataset=dataset, trial=trial, fold=fold, recommender=recommender,
            tradeoff=tradeoff, distribution=distribution, fairness=calibration,
            relevance=relevance, tradeoff_weight=weight, select_item=selector
    ):
        logger.info(">> Already Done... " + system_name)
        return "Already Done"
    try:
        clock = Clocker()
        # Starting the counter
        clock.start_count()

        # Instancing the post-processing
        pierre = PostProcessingStep(
            recommender=recommender, dataset_name=dataset, trial=trial, fold=fold,
            tradeoff_component=tradeoff, distribution_component=distribution,
            fairness_component=calibration, relevance_component=relevance,
            tradeoff_weight_component=weight, selector_component=selector,
            list_size=list_size, alpha=alpha, d=d
        )
        logger.info(">> Running... " + system_name)
        pierre.run()

        # Finishing the counter
        clock.finish_count()

        # Saving execution time
        SaveAndLoad.save_postprocessing_time(
            data=clock.clock_data(),
            dataset=dataset, trial=trial, fold=fold, recommender=recommender,
            tradeoff=tradeoff, distribution=distribution, fairness=calibration,
            relevance=relevance, tradeoff_weight=weight, select_item=selector
        )

        # Finishing the step
        clock.print_time_info()
        return "Finished"
    except Exception as e:
        logger.error(">> Error... " + system_name)
        logger.exception(e)
        return "Error"


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

    def print_basic_info(self) -> None:
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
        logger.info("$" * 50)
        logger.info("$" * 50)

    def main(self) -> None:
        combination = [
            self.experimental_settings['recommender'],
            self.experimental_settings['tradeoff'], self.experimental_settings['relevance'],
            self.experimental_settings['distribution'], self.experimental_settings['selector'],
            self.experimental_settings['weight'], self.experimental_settings['fairness'],
            self.experimental_settings['list_size'], self.experimental_settings['alpha'],
            self.experimental_settings['d'], [self.experimental_settings["checkpoint"]],
            self.experimental_settings['dataset'],
            self.experimental_settings['fold'], self.experimental_settings['trial'],
        ]

        if self.experimental_settings['multiprocessing'] == "joblib":
            load = Parallel(n_jobs=self.experimental_settings["n_jobs"], backend="multiprocessing", prefer="processes",
                            verbose=10, batch_size=1)(
                delayed(starting_postprocessing)(
                    recommender=recommender, dataset=dataset, trial=trial, fold=fold,
                    tradeoff=tradeoff, relevance=relevance, distribution=distribution,
                    selector=selector, weight=weight, calibration=calibration,
                    list_size=list_size, alpha=alpha, d=d, checkpoint=checkpoint
                ) for
                recommender, tradeoff, relevance, distribution, selector, weight, calibration, list_size, alpha, d, checkpoint, dataset, fold, trial
                in list(itertools.product(*combination))
            )
        elif self.experimental_settings['multiprocessing'] == "starmap":
            process_args = []
            for recommender, tradeoff, relevance, distribution, selector, weight, calibration, list_size, alpha, d, checkpoint, dataset, fold, trial in list(
                    itertools.product(*combination)):
                process_args.append((
                    recommender, fold, trial, dataset, tradeoff, distribution, calibration, relevance, weight, selector,
                    list_size, alpha, d, checkpoint))
            pool = multiprocessing.Pool(processes=self.experimental_settings["n_jobs"])
            load = pool.starmap(starting_postprocessing, process_args)
            pool.close()
            pool.join()
        else:
            logger.warning(
                f"The multiprocessing option {self.experimental_settings['multiprocessing']} does not exist! Please check for a possible option.")
            exit(1)

        jobs = dict(Counter(load))
        logger.info(jobs)


if __name__ == '__main__':
    """
    It starts the post-processing step
    """
    logger.info(" ".join(['+' * 10, 'System Starting', '+' * 10]))
    step = PierreStep4()
    step.read_the_entries()
    step.print_basic_info()
    step.main()
    logger.info(" ".join(['+' * 10, 'System shutdown', '+' * 10]))
