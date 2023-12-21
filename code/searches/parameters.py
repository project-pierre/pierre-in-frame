from scipy.stats import randint, uniform


class SurpriseParams:
    size = 10
    # 1: User KNN
    USER_KNN_SEARCH_PARAMS = {"k": randint(3, 203).rvs(size=size),
                              "sim_options": {'name': ['pearson', 'cosine', 'msd'], 'user_based': [True]}}
    # 2: Item KNN
    ITEM_KNN_SEARCH_PARAMS = {"k": randint(3, 201).rvs(size=size),
                              "sim_options": {'name': ['pearson', 'cosine', 'msd'], 'user_based': [False]}}
    # 3: SVD
    SVD_SEARCH_PARAMS = {"n_factors": randint(10, 150).rvs(size=size), "n_epochs": randint(10, 150).rvs(size=size),
                         "lr_all": uniform(0.001, 0.01).rvs(size=size), "reg_all": uniform(0.01, 0.1).rvs(size=size)}
    # 4: SVD++
    SVDpp_SEARCH_PARAMS = {"n_factors": randint(10, 150).rvs(size=size), "n_epochs": randint(10, 150).rvs(size=size),
                           "lr_all": uniform(0.001, 0.01).rvs(size=size), "reg_all": uniform(0.01, 0.1).rvs(size=size)}
    # 5: NMF
    NMF_SEARCH_PARAMS = {"n_factors": randint(10, 150).rvs(size=size), "n_epochs": randint(10, 150).rvs(size=size),
                         "reg_pu": uniform(0.01, 0.1).rvs(size=size), "reg_qi": uniform(0.01, 0.1).rvs(size=size),
                         "reg_bu": uniform(0.01, 0.1).rvs(size=size), "reg_bi": uniform(0.01, 0.1).rvs(size=size),
                         "lr_bu": uniform(0.001, 0.01).rvs(size=size), "lr_bi": uniform(0.001, 0.01).rvs(size=size),
                         "biased": [True]}
    # 6: Co Clustering
    CLUSTERING_SEARCH_PARAMS = {"n_cltr_u": randint(3, 110).rvs(size=size), "n_cltr_i": randint(3, 110).rvs(size=size),
                                "n_epochs": randint(10, 150).rvs(size=size)}


class ConformityParams:
    # Cluster Params
    CLUSTER_PARAMS = {
        'n_clusters': [
            2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97
        ],
    }

    # Cluster Grid Search
    CLUSTER_PARAMS_GRID = {
        'min_samples': [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37],
        'eps': [
            0.05, 0.10, 0.15, 0.20, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55,
        ],
        'metric': [
            'cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan',
            'braycurtis', 'canberra', 'chebyshev', 'correlation', 'hamming'
        ]
    }

    COMPONENT_PARAMS_GRID = {
        'n_components': [
            1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97
        ]
    }

    ESTIMATORS_PARAMS_GRID = {
        'n_estimators': [
            1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97
        ]
    }

    NEIGHBOR_PARAMS_GRID = {
        'n_neighbors': [
            1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97
        ],
        'metric': [
            'cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan',
            'braycurtis', 'canberra', 'chebyshev', 'correlation', 'hamming'
        ]
    }

    OUTLIEAR_PARAMS_GRID = {
        'nu': [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    }
