import itertools
import os

import pandas as pd
import numpy as np

from datasets.utils.base import Dataset
from settings.constants import Constants
from settings.labels import Label
from settings.path_dir_file import PathDirFile


class LastFMTwoBillion(Dataset):
    """
    Last FM 2 Billion dataset.
    This class organizes the work with the dataset.
    """
    # Class information.
    dir_name = "lfm-2b"
    verbose_name = "Last FM Two Billion"
    system_name = "lfm-2b"

    # Raw paths.
    dataset_raw_path = "/".join([PathDirFile.RAW_DATASETS_DIR, dir_name])
    raw_transaction_file = "listening-counts.tsv"
    raw_transaction_events_file = "listening_events.tsv"
    raw_items_file = "tags-micro-genres.json"

    # Clean paths.
    dataset_clean_path = "/".join([PathDirFile.CLEAN_DATASETS_DIR, dir_name])

    # Constant Values
    cut_value = 2

    # ######################################### #
    # ############## Constructor ############## #
    # ######################################### #

    def __init__(self):
        """
        Class constructor. Firstly call the super constructor and after start personalized things.
        """
        super().__init__()
        self.raw_transactions_events = None

    # ######################################### #
    # ############# Transactions ############## #
    # ######################################### #

    def load_raw_transactions(self):
        """
        Load raw transactions into the instance variable.
        """
        self.raw_transactions = pd.read_csv(
            os.path.join(self.dataset_raw_path, self.raw_transaction_file),
            names=['user_id', 'track_id', 'count'],
            engine='python', sep='\t'
        )
        self.raw_transactions_events = pd.read_csv(
            os.path.join(self.dataset_raw_path, self.raw_transaction_events_file),
            names=['user_id', 'track_id', 'album_id', 'timestamp'],
            engine='python', sep='\t'
        )

    def get_raw_transactions_events(self):
        if self.raw_transactions_events is None:
            self.load_raw_transactions()
        return self.raw_transactions_events

    def clean_transactions(self):
        """
        Cleaning the raw transactions and save as clean transactions.
        """
        super().clean_transactions()

        # Load the raw transactions.
        raw_transactions = self.get_raw_transactions().sort_values(by=['user_id', 'track_id'], ascending=False)
        raw_transactions_events = self.get_raw_transactions_events().drop_duplicates(subset=['user_id', 'track_id'], keep='last')
        raw_transactions_events.sort_values(by=['user_id', 'track_id'], ascending=False, inplace=True)
        raw_transactions['timestamp'] = raw_transactions_events['timestamp'].to_list()
        self.transactions = raw_transactions

        # Filter transactions based on the items id list.
        # filtered_raw_transactions = raw_transactions[
        #     raw_transactions[Label.ITEM_ID].isin(self.items[Label.ITEM_ID].tolist())]

        # Cut users and set the new data into the instance.
        # self.set_transactions(
        #     new_transactions=LastFMTwoBillion.cut_users(raw_transactions, self.cut_value))
        #
        # if Constants.NORMALIZED_SCORE:
        #     self.transactions[Label.TRANSACTION_VALUE] = np.where(self.transactions[Label.TRANSACTION_VALUE] >= self.cut_value, 1, 0)

        # Save the clean transactions as CSV.
        self.transactions.to_csv(
            os.path.join(self.dataset_clean_path, PathDirFile.TRANSACTIONS_FILE),
            index=False
        )

    # ######################################### #
    # ################# Items ################# #
    # ######################################### #

    def load_raw_items(self):
        """
        Load Raw Items into the instance variable.
        """
        self.raw_items = pd.read_csv(
            os.path.join(self.dataset_raw_path, self.raw_items_file), engine='python',
            sep='::', names=[Label.ITEM_ID, Label.TITLE, Label.GENRES], encoding='ISO-8859-1'
        )

    def clean_items(self):
        """
        Cleaning the raw items and save as clean items.
        """
        # Load the raw items.
        raw_items_df = self.get_raw_items()

        # Clean the items without information and with the label indicating no genre in the item.
        raw_items_df.dropna(inplace=True)
        genre_clean_items = raw_items_df[raw_items_df[Label.GENRES] != '(no genres listed)']

        # Set the new data into the instance.
        self.set_items(new_items=genre_clean_items)
        self.items.drop_duplicates(subset=[Label.ITEM_ID], inplace=True)

        # Save the clean transactions as CSV.
        self.items.to_csv(os.path.join(self.dataset_clean_path, PathDirFile.ITEMS_FILE), index=False)
