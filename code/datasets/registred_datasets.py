from datasets.last_fm_two_billion import LastFMTwoBillion
from datasets.movielens_one_million import MovielensOneMillion
from datasets.my_anime_list import MyAnimeList
from datasets.taste_profile import TasteProfile
from datasets.twitter_movies import TwitterMovies
from datasets.utils.base import Dataset
from datasets.yahoo_movies import YahooMovies


class RegisteredDataset:
    """
    All datasets allowed to the system use, are registered in this class.
    """
    # Registered datasets
    MOVIELENS_ONE_MILLION_DATASET = MovielensOneMillion.system_name
    YAHOO_MOVIES_DATASET = YahooMovies.system_name
    TASTE_PROFILE_DATASET = TasteProfile.system_name
    MY_ANIME_LIST_DATASET = MyAnimeList.system_name
    TWITTER_MOVIES_DATASET = TwitterMovies.system_name
    LASTFM_TWO_BILLION_DATASET = LastFMTwoBillion.system_name

    # Allowed to be accessed
    DATASET_LIST = [
        MOVIELENS_ONE_MILLION_DATASET, YAHOO_MOVIES_DATASET, TWITTER_MOVIES_DATASET,
        LASTFM_TWO_BILLION_DATASET,
        # TASTE_PROFILE_DATASET,
        # MY_ANIME_LIST_DATASET
    ]

    # Default dataset
    DEFAULT_DATASET = YAHOO_MOVIES_DATASET

    @staticmethod
    def load_dataset(dataset: str) -> Dataset:
        """
        Load the dataset instance, if the dataset is registered.
        :param dataset: A string with the system name of the dataset.
        :return: A dataset class instance.
        """
        # Movielens One Million
        if dataset == RegisteredDataset.MOVIELENS_ONE_MILLION_DATASET:
            return MovielensOneMillion()
        # Yahoo Movies
        elif dataset == RegisteredDataset.YAHOO_MOVIES_DATASET:
            return YahooMovies()
        # Taste Profile
        elif dataset == RegisteredDataset.TASTE_PROFILE_DATASET:
            return TasteProfile()
        # My Anime List
        elif dataset == RegisteredDataset.MY_ANIME_LIST_DATASET:
            return MyAnimeList()
        # Twitter Movies
        elif dataset == RegisteredDataset.TWITTER_MOVIES_DATASET:
            return TwitterMovies()
        # Last FM Two Billion
        elif dataset == RegisteredDataset.LASTFM_TWO_BILLION_DATASET:
            return LastFMTwoBillion()
        else:
            raise "The requested dataset is not registered in the system"

    @staticmethod
    def preprocessing(dataset: str, n_trials: int, n_folds: int):
        """
        Preprocessing the requested dataset, if the dataset is registered.
        :param dataset: A string wih the system name of the dataset.
        :param n_trials: An int that represents a number of the experimental trial.
        :param n_folds: An int that represents a number of the k folds.
        """
        # Movielens One Million
        if dataset == RegisteredDataset.MOVIELENS_ONE_MILLION_DATASET:
            instance = MovielensOneMillion()
            instance.mining_data_and_create_fold(n_trials=n_trials, n_folds=n_folds)
        # Yahoo Movies
        elif dataset == RegisteredDataset.YAHOO_MOVIES_DATASET:
            instance = YahooMovies()
            instance.mining_data_and_create_fold(n_trials=n_trials, n_folds=n_folds)
        # Taste Profile
        elif dataset == RegisteredDataset.TASTE_PROFILE_DATASET:
            instance = TasteProfile()
            instance.mining_data_and_create_fold(n_trials=n_trials, n_folds=n_folds)
        # My Anime List
        elif dataset == RegisteredDataset.MY_ANIME_LIST_DATASET:
            instance = MyAnimeList()
            instance.mining_data_and_create_fold(n_trials=n_trials, n_folds=n_folds)
        # Twitter Movies
        elif dataset == RegisteredDataset.TWITTER_MOVIES_DATASET:
            instance = TwitterMovies()
            instance.mining_data_and_create_fold(n_trials=n_trials, n_folds=n_folds)
        # Last FM Two Billion
        elif dataset == RegisteredDataset.LASTFM_TWO_BILLION_DATASET:
            instance = LastFMTwoBillion()
            instance.mining_data_and_create_fold(n_trials=n_trials, n_folds=n_folds)
        else:
            raise "The requested dataset is not registered in the system"
