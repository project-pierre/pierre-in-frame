import itertools
import logging
import multiprocessing
from joblib import Parallel, delayed

from checkpoint_verification import CheckpointVerification
from evaluations import evaluation_interface
from evaluations.conformity_algorithms import ConformityAlgorithms
from settings.labels import Label
from settings.path_dir_file import PathDirFile
from settings.save_and_load import SaveAndLoad
from utils.clocker import Clocker
from utils.input import Input
from utils.logging_settings import setup_logging
from utils.step import Step

logger = logging.getLogger(__name__)


def applying_evaluation_metrics(
        metrics: list, recommender: str, dataset: str, trial: int, fold: int,
        distribution: str, fairness: str, relevance: str, weight: str, tradeoff: str, selector: str,
        checkpoint: str
):
    """
    Function to apply the evaluation metrics.
    """
    load_monitoring = []
    for m in metrics:
        if m == Label.MAP:
            load_monitoring.append(
                evaluation_interface.applying_map(
                    recommender=recommender, dataset=dataset, trial=trial, fold=fold,
                    distribution=distribution, fairness=fairness, relevance=relevance,
                    weight=weight, tradeoff=tradeoff, selector=selector, checkpoint=checkpoint
                )
            )
        elif m == Label.MRR:
            load_monitoring.append(
                evaluation_interface.applying_mrr(
                    recommender=recommender, dataset=dataset, trial=trial, fold=fold,
                    distribution=distribution, fairness=fairness, relevance=relevance,
                    weight=weight, tradeoff=tradeoff, selector=selector, checkpoint=checkpoint
                )
            )
        elif m == Label.MACE:
            load_monitoring.append(
                evaluation_interface.applying_mace(
                    recommender=recommender, dataset=dataset, trial=trial, fold=fold,
                    distribution=distribution, fairness=fairness, relevance=relevance,
                    weight=weight, tradeoff=tradeoff, selector=selector, checkpoint=checkpoint
                )
            )
        elif m == Label.MRMC:
            load_monitoring.append(
                evaluation_interface.applying_mrmc(
                    recommender=recommender, dataset=dataset, trial=trial, fold=fold,
                    distribution=distribution, fairness=fairness, relevance=relevance,
                    weight=weight, tradeoff=tradeoff, selector=selector, checkpoint=checkpoint
                )
            )
        elif m == Label.UNEXPECTEDNESS:
            load_monitoring.append(
                evaluation_interface.applying_unexpectedness(
                    recommender=recommender, dataset=dataset, trial=trial, fold=fold,
                    distribution=distribution, fairness=fairness, relevance=relevance,
                    weight=weight, tradeoff=tradeoff, selector=selector, checkpoint=checkpoint
                )
            )
        elif m == Label.SERENDIPITY:
            load_monitoring.append(
                evaluation_interface.applying_serendipity(
                    recommender=recommender, dataset=dataset, trial=trial, fold=fold,
                    distribution=distribution, fairness=fairness, relevance=relevance,
                    weight=weight, tradeoff=tradeoff, selector=selector, checkpoint=checkpoint
                )
            )
        elif m == Label.NOVELTY:
            load_monitoring.append(
                evaluation_interface.applying_novelty(
                    recommender=recommender, dataset=dataset, trial=trial, fold=fold,
                    distribution=distribution, fairness=fairness, relevance=relevance,
                    weight=weight, tradeoff=tradeoff, selector=selector, checkpoint=checkpoint
                )
            )
        elif m == Label.COVERAGE:
            load_monitoring.append(
                evaluation_interface.applying_coverage(
                    recommender=recommender, dataset=dataset, trial=trial, fold=fold,
                    distribution=distribution, fairness=fairness, relevance=relevance,
                    weight=weight, tradeoff=tradeoff, selector=selector, checkpoint=checkpoint
                )
            )
        elif m == Label.PERSONALIZATION:
            load_monitoring.append(
                evaluation_interface.applying_personalization(
                    recommender=recommender, dataset=dataset, trial=trial, fold=fold,
                    distribution=distribution, fairness=fairness, relevance=relevance,
                    weight=weight, tradeoff=tradeoff, selector=selector, checkpoint=checkpoint
                )
            )
        elif m == "TIME":
            load_monitoring.append(
                evaluation_interface.execution_time_fold(
                    recommender=recommender, dataset=dataset, trial=trial, fold=fold,
                    distribution=distribution, fairness=fairness, relevance=relevance,
                    weight=weight, tradeoff=tradeoff, selector=selector, checkpoint=checkpoint
                )
            )
        else:
            continue
    return load_monitoring


def starting_cluster(
        cluster: str, recommender: str, dataset: str, trial: int, fold: int,
        distribution: str, fairness: str, relevance: str, weight: str, tradeoff: str, selector: str,
        checkpoint: str
):
    """
    TODO
    """
    # self.set_the_logfile_by_instance(
    #         recommender=recommender, dataset=dataset, trial=trial, fold=fold,
    #         distribution=distribution, fairness=fairness, relevance=relevance,
    #         tradeoff_weight=weight, tradeoff=tradeoff, select_item=selector
    # )

    system_name = "-".join([
        dataset, 'trial-' + str(trial), 'fold-' + str(fold), recommender,
        tradeoff, distribution, relevance, selector, fairness, tradeoff
    ])

    if checkpoint == "YES" and CheckpointVerification.unit_step5_conformity_verification(
            dataset=dataset, trial=trial, fold=fold,
            cluster=cluster, metric=Label.JACCARD_SCORE, recommender=recommender,
            distribution=distribution, fairness=fairness, relevance=relevance,
            weight=weight, tradeoff=tradeoff, selector=selector
    ):
        logger.info(">> Already Done... " + system_name)
        return "Already Done"

    clock = Clocker()
    # Starting the counter
    clock.start_count()

    # Executing the Random Search
    cluster_instance = ConformityAlgorithms(
        cluster=cluster,
        recommender=recommender, dataset=dataset, trial=trial, fold=fold,
        distribution=distribution, fairness=fairness, relevance=relevance,
        weight=weight, tradeoff=tradeoff, selector=selector
    )
    cluster_instance.prepare_experiment()
    cluster_instance.fit()

    cluster_instance.evaluation()

    # Finishing the counter
    clock.finish_count()

    # Saving execution time
    SaveAndLoad.save_conformity_metric_time(
        data=clock.clock_data(), cluster=cluster,
        recommender=recommender, dataset=dataset, trial=trial, fold=fold,
        distribution=distribution, fairness=fairness, relevance=relevance,
        weight=weight, tradeoff=tradeoff, selector=selector
    )


class PierreStep5(Step):
    """
    TODO: Docstring
    """

    def read_the_entries(self):
        """
        TODO: Docstring
        """
        self.experimental_settings = Input.step5()

    @staticmethod
    def set_the_logfile_by_instance(
            dataset: str, recommender: str, trial: int, fold: int, tradeoff: str, distribution: str,
            fairness: str, relevance: str, tradeoff_weight: str, select_item: str
    ):
        """
        TODO: Docstring
        """
        # Setup Log configuration
        setup_logging(
            log_error="error.log", log_info="info.log",
            save_path=PathDirFile.set_log_metrics_path(
                dataset=dataset, recommender=recommender, trial=trial, fold=fold, tradeoff=tradeoff,
                distribution=distribution, fairness=fairness, relevance=relevance, tradeoff_weight=tradeoff_weight,
                select_item=select_item
            )
        )

    def print_basic_info_by_instance(self, **kwargs):
        """
        TODO: Docstring
        """

        logger.info("$" * 50)
        logger.info("$" * 50)
        # Logging machine data
        self.machine_information()
        logger.info("-" * 50)

        # Logging the experiment setup
        logger.info("[METRIC STEP]")
        logger.info(kwargs)
        logger.info("$" * 50)
        logger.info("$" * 50)

    def main(self):
        """
        TODO: Docstring
        """
        if self.experimental_settings['opt'] == Label.CONFORMITY:
            self.cluster_parallelization()
        elif self.experimental_settings['opt'] == Label.EVALUATION_METRICS:
            self.metrics_parallelization()
        else:
            print(f"Option {self.experimental_settings['opt']} is not registered!")

    def metrics_parallelization(self):
        combination = [
            self.experimental_settings['recommender'], self.experimental_settings['dataset'],
            self.experimental_settings['trial'], self.experimental_settings['fold'],
            self.experimental_settings['distribution'], self.experimental_settings['fairness'],
            self.experimental_settings['relevance'], self.experimental_settings['weight'],
            self.experimental_settings['tradeoff'], self.experimental_settings['selector'],
            [self.experimental_settings["checkpoint"]]
        ]
        process_combination = list(itertools.product(*combination))
        print(f"The total of process that will be run are: {len(process_combination)}")
        if self.experimental_settings['multiprocessing'] == "joblib":
            Parallel(
                n_jobs=self.experimental_settings['n_jobs'], verbose=100,
                backend="multiprocessing", prefer="processes"
            )(
                delayed(applying_evaluation_metrics)(
                    metrics=self.experimental_settings['metric'],
                    recommender=recommender, dataset=dataset, trial=trial, fold=fold,
                    distribution=distribution, fairness=fairness, relevance=relevance,
                    weight=weight, tradeoff=tradeoff, selector=selector, checkpoint=checkpoint
                ) for
                recommender, dataset, trial, fold, distribution, fairness, relevance, weight, tradeoff, selector, checkpoint
                in process_combination
            )
        elif self.experimental_settings['multiprocessing'] == "starmap":
            process_args = []
            for recommender, tradeoff, relevance, distribution, selector, weight, calibration, list_size, alpha, d, checkpoint, dataset, fold, trial in process_combination:
                process_args.append((
                    recommender, fold, trial, dataset, tradeoff, distribution, calibration, relevance, weight, selector,
                    list_size, alpha, d, checkpoint
                ))
            pool = multiprocessing.Pool(processes=self.experimental_settings["n_jobs"])
            pool.starmap(applying_evaluation_metrics, process_args)
            pool.close()
            pool.join()
        else:
            logger.warning(
                f"The multiprocessing option {self.experimental_settings['multiprocessing']} does not exist! Please check for a possible option.")
            exit(1)

    def cluster_parallelization(self):
        """
        TODO
        """
        combination = [
            self.experimental_settings['cluster'],
            self.experimental_settings['recommender'], self.experimental_settings['dataset'],
            self.experimental_settings['trial'], self.experimental_settings['fold'],
            self.experimental_settings['distribution'], self.experimental_settings['fairness'],
            self.experimental_settings['relevance'], self.experimental_settings['weight'],
            self.experimental_settings['tradeoff'], self.experimental_settings['selector'],
            [self.experimental_settings["checkpoint"]]
        ]
        print(f"The total of process that will be run are: {len(combination)}")

        Parallel(n_jobs=self.experimental_settings['n_jobs'])(
            delayed(starting_cluster)(
                cluster=cluster,
                recommender=recommender, dataset=dataset, trial=trial, fold=fold,
                distribution=distribution, fairness=fairness, relevance=relevance,
                weight=weight, tradeoff=tradeoff, selector=selector, checkpoint=checkpoint
            ) for
            cluster, recommender, dataset, trial, fold, distribution, fairness, relevance, weight, tradeoff, selector, checkpoint
            in list(itertools.product(*combination))
        )


if __name__ == '__main__':
    """
    It starts the metric step
    """
    logger.info(" ".join(['+' * 10, 'System Starting', '+' * 10]))
    step = PierreStep5()
    step.read_the_entries()
    step.main()
    logger.info(" ".join(['+' * 10, 'System shutdown', '+' * 10]))
