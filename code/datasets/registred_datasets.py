from datasets.foodcom import FoodComRecipe
from datasets.last_fm_two_billion import LastFMTwoBillion
from datasets.movielens_one_million import MovielensOneMillion
from datasets.movielens_twenty_million import MovielensTwentyMillion
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
    MOVIELENS_TWENTY_MILLION_DATASET = MovielensTwentyMillion.system_name
    FOOD_COM_RECIPE_DATASET = FoodComRecipe.system_name
    YAHOO_MOVIES_DATASET = YahooMovies.system_name
    TASTE_PROFILE_DATASET = TasteProfile.system_name
    MY_ANIME_LIST_DATASET = MyAnimeList.system_name
    TWITTER_MOVIES_DATASET = TwitterMovies.system_name
    LASTFM_TWO_BILLION_DATASET = LastFMTwoBillion.system_name

    # Allowed to be accessed
    DATASET_LIST = [
        MOVIELENS_ONE_MILLION_DATASET, YAHOO_MOVIES_DATASET, TWITTER_MOVIES_DATASET,
        LASTFM_TWO_BILLION_DATASET, MOVIELENS_TWENTY_MILLION_DATASET, FOOD_COM_RECIPE_DATASET
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
        # Movielens Twenty Million Movies
        elif dataset == RegisteredDataset.MOVIELENS_TWENTY_MILLION_DATASET:
            return MovielensTwentyMillion()
        # Food.com recipes
        elif dataset == RegisteredDataset.FOOD_COM_RECIPE_DATASET:
            return FoodComRecipe()
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
    def preprocessing(
            dataset: str, n_trials: int, n_folds: int,
            cut_value: int, item_cut_value: int, profile_len_cut_value: int
    ):
        """
        Preprocessing the requested dataset, if the dataset is registered.
        :param dataset: A string wih the system name of the dataset.
        :param n_trials: An int that represents a number of the experimental trial.
        :param n_folds: An int that represents a number of the k folds.
        """
        # Movielens One Million
        if dataset == RegisteredDataset.MOVIELENS_ONE_MILLION_DATASET:
            instance = MovielensOneMillion()
            instance.set_experiment_variables(cut_value=cut_value, item_cut_value=item_cut_value, profile_len_cut_value=profile_len_cut_value)
            instance.mining_data_and_create_fold(n_trials=n_trials, n_folds=n_folds)
        # Movielens Twenty Millions Movies
        elif dataset == RegisteredDataset.MOVIELENS_TWENTY_MILLION_DATASET:
            instance = MovielensTwentyMillion()
            instance.set_experiment_variables(cut_value=cut_value, item_cut_value=item_cut_value, profile_len_cut_value=profile_len_cut_value)
            instance.mining_data_and_create_fold(n_trials=n_trials, n_folds=n_folds)
        # Food.com Recipe
        elif dataset == RegisteredDataset.FOOD_COM_RECIPE_DATASET:
            instance = FoodComRecipe()
            instance.set_experiment_variables(cut_value=cut_value, item_cut_value=item_cut_value, profile_len_cut_value=profile_len_cut_value)
            instance.mining_data_and_create_fold(n_trials=n_trials, n_folds=n_folds)
        # Yahoo Movies
        elif dataset == RegisteredDataset.YAHOO_MOVIES_DATASET:
            instance = YahooMovies()
            instance.set_experiment_variables(cut_value=cut_value, item_cut_value=item_cut_value, profile_len_cut_value=profile_len_cut_value)
            instance.mining_data_and_create_fold(n_trials=n_trials, n_folds=n_folds)
        # Taste Profile
        elif dataset == RegisteredDataset.TASTE_PROFILE_DATASET:
            instance = TasteProfile()
            instance.set_experiment_variables(cut_value=cut_value, item_cut_value=item_cut_value, profile_len_cut_value=profile_len_cut_value)
            instance.mining_data_and_create_fold(n_trials=n_trials, n_folds=n_folds)
        # My Anime List
        elif dataset == RegisteredDataset.MY_ANIME_LIST_DATASET:
            instance = MyAnimeList()
            instance.set_experiment_variables(cut_value=cut_value, item_cut_value=item_cut_value, profile_len_cut_value=profile_len_cut_value)
            instance.mining_data_and_create_fold(n_trials=n_trials, n_folds=n_folds)
        # Twitter Movies
        elif dataset == RegisteredDataset.TWITTER_MOVIES_DATASET:
            instance = TwitterMovies()
            instance.set_experiment_variables(cut_value=cut_value, item_cut_value=item_cut_value, profile_len_cut_value=profile_len_cut_value)
            instance.mining_data_and_create_fold(n_trials=n_trials, n_folds=n_folds)
        # Last FM Two Billion
        elif dataset == RegisteredDataset.LASTFM_TWO_BILLION_DATASET:
            instance = LastFMTwoBillion()
            instance.set_experiment_variables(cut_value=cut_value, item_cut_value=item_cut_value, profile_len_cut_value=profile_len_cut_value)
            instance.mining_data_and_create_fold(n_trials=n_trials, n_folds=n_folds)
        else:
            raise "The requested dataset is not registered in the system"
