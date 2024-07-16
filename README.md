# Pierre-in-Frame
## It is a frozen code for the paper "Considering Time and Feature Entropy in Calibrated Recommendations" authored by Diego Corrêa da Silva, Dietmar Jannach and Frederico Araújo Durão.

[Pre-Print](https://readthedocs.io/) | [Paper](https://doi.org/xx.xxx/xxxxx.xxxxxx)

[![Version](https://img.shields.io/badge/version-v0.2-green)](https://github.com/project-pierre/pierre-in-frame) ![PyPI - Python Version](https://img.shields.io/badge/python-3.8-blue)

[![logo](MethodologyWorkflow.svg)](https://project-pierre.github.io/project-pierre)

Pierre-in-Frame is a how-to-create recommendation system focused on multi-objectives.
The framework is a combination of other libraries on recommendation systems and machine learning, such as Scikit-Surprise, Scikit-Pierre and Scikit-learn.
It is constructed to be totally focused on providing a way to produce recommendation systems with more than one objective as calibration, diversity, etc.
In the version 0.1, the framework implements seven collaborative filtering algorithms on the processing and the concept of calibration on the post-processing.
Pierre-in-Frame are composed by **seven steps** in sequence, applying the concept of auto-recsys (auto-ml).
It is totally necessary to run each step following the initial experiment definition because of the concept of auto-recsys.

To create an experiment each step has parameters that receive options from the command line.
Along the steps, the personalization of these parameters can create more than 40 thousand recommendation systems, providing different performance results to the same dataset.
Be cautious with the parameters, if you badly define the configuration a high number of recommender systems will be created, and it is put in a line.

# How-To-Use

## Preparing  
1. Update and upgrade the OS: `sudo apt update && sudo apt upgrade -y && sudo apt install curl`  
2. Download the Anaconda: `curl -O https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-x86_64.sh`   
3. Install the Anaconda: `bash Anaconda3-2023.09-0-Linux-x86_64.sh`  
 
## The code
1. Download the framework: `git clone git@github.com:project-pierre/pierre-in-frame.git`  
2. Go to the framework path: `cd pierre-in-frame/`
3. The framework structure: `ls .`  
3.1. `pierre_in_frame/`: Where all python code is. It is the kernel of the framework code.  
3.2. `data/`: Where all data are saved as raw and clean datasets, candidate items, recommendation lists, execution time, etc.  
3.3. `docs/`: The documentation dir to be used on the readthedocs. **TODO: It is a future work**.  
3.4. `environment/`: It is used to save all framework environment files. The files have the configuration of the run. Each step has a different running variable.  
3.5. `logs/`: It is produced after the first run. The log path is used to save the code status. PS: This directory is created after the first run.  
3.6. `metadata/`: Metadata about the framework.  
3.7. `results/`: The result files will be saved inside this dir, tables, figures, etc.  
3.8. `shell/`: Examples of how-to-run on Linux shell.  

## Conda Environment
1. Load Conda Environment: `conda env create -f environment.yml`    
2. Start the conda env: `conda activate pierre-in-frame`  
3. Install the framework environment: `conda env create -f environment.yml`  
4. Activate the environment: `conda activate pierre-in-frame`  
   4.1. It is also possible to install it through pip requirements  
5. Download, compile, and install the `Project Pierre` libraries
   5.1. Download the Scikit-Pierre, compile, and install it: `https://github.com/project-pierre/scikit_pierre`  
   5.2. Download the Recommender-Pierre, compile, and install it: `https://github.com/project-pierre/recommender-pierre`

## Using the framework
To run the framework it is necessary to go to dir: `cd pierre_in_frame/`. 
Inside this directory are seven python scripts. Each step has a set of parameters that are necessary to run.    

### Step 1: Pre-processing
The pre-processing step is implemented for cleaning, filtering and modeling the data.
The framework pre-processing the dataset to provide a structure to be used by all next steps.
It applies a data partition by your choice. We have implemented four pre-process validation methods.
The pre-processing script provides three functionality to config the experiment:
1. Pre-process based on validation method  
   1.1. Movielens 20M: `python3 step1_preprocessing.py from_file=YES file_name=ml-20m`  
   1.2. Food.com: `python3 step1_preprocessing.py from_file=YES file_name=food`  
   1.3. Last.fm 2B: `python3 step1_preprocessing.py from_file=YES file_name=lfm-2b-subset`  
2. Pre-Compute Distribution. It improves the execution time.  
   2.1. Movielens 20M: `python3 step1_preprocessing.py from_file=YES file_name=ml-20m_distribution`  
   2.2. Food.com: `python3 step1_preprocessing.py from_file=YES file_name=food_distribution`  
   2.3. Last.fm 2B: `python3 step1_preprocessing.py from_file=YES file_name=lfm-2b-subset_distribution`  
3. Pre-Compute One-Hot-Encode. It improves the execution time.  
   3.1. Movielens 20M: `python3 step1_preprocessing.py from_file=YES file_name=ml-20m_ohe`  
   3.2. Food.com: `python3 step1_preprocessing.py from_file=YES file_name=food_ohe`  
   3.3. Last.fm 2B: `python3 step1_preprocessing.py from_file=YES file_name=lfm-2b-subset_ohe`  

You can find the environment files from step 1 in the directory `environment/step1/`.

#### Run Examples
In the directory `shell`, there are examples of script to run the step 1 for the three dataset cited above.
From the main project directory, you can run the command: `sh shell/step1.sh > ./logs/step1.log 2>&1 & disown`.
It will run all step 1 commands showed above.

### Step 2: Best Parameter Search
In this step, the Random Search method is applied to find the best set of hyperparameters.
The best set of parameter values are saved inside the path `data/experiment/hyperparameters/<dataset_name>/`.

#### Run Examples
1. Movielens 20M   
   1.1. CVTT + BPR + ALS: `python3 step2_random_search.py from_file=YES file_name=ml-20m`    
   1.2. CVTT + DeepAE: `python3 step2_random_search.py from_file=YES file_name=deep_ae_ml-20m`    
2. Food.com Recipes    
   1.1. CVTT + BPR + ALS: `python3 step2_random_search.py from_file=YES file_name=food`    
   1.2. CVTT + DeepAE: `python3 step2_random_search.py from_file=YES file_name=deep_ae_food`     
3. Last.fm    
   1.1. CVTT + BPR + ALS: `python3 step2_random_search.py from_file=YES file_name=lfm-2b-subset`    
   1.2. CVTT + DeepAE: `python3 step2_random_search.py from_file=YES file_name=deep_ae_lfm-2b-subset`    

In the directory `shell`, there are examples of script to run the step 2 for the three dataset cited above.
From the main project directory, you can run the command: `sh shell/step2.sh > ./logs/step2.log 2>&1 & disown`.
It will run all step 2 commands showed above.

### Step 3: Processing
The processing step uses the data pre-processed and the hyperparameters to train the chosen recommender algorithm.
The prediction generates a set of candidate items to be used by the post-processing.
All candidate items set are saved inside `data/experiment/{dataset}/candidate_items/{recommender}/trial-{trial}/fold-{fold}/candidate_items.csv`.

#### Run Examples

1. Movielens 20M   
   1.1. BPR + ALS: `python3 step3_processing.py from_file=YES file_name=ml-20m`    
   1.2. DeepAE: `python3 step3_processing.py from_file=YES file_name=deep_ae_ml-20m`    
2. Food.com Recipes    
   1.1. BPR + ALS: `python3 step3_processing.py from_file=YES file_name=food`    
   1.2. DeepAE: `python3 step3_processing.py from_file=YES file_name=deep_ae_food`     
3. Last.fm    
   1.1. BPR + ALS: `python3 step3_processing.py from_file=YES file_name=lfm-2b-subset`    
   1.2. DeepAE: `python3 step3_processing.py from_file=YES file_name=deep_ae_lfm-2b-subset`

In the directory `shell`, there are examples of script to run the step 3 for the three dataset cited above.
From the main project directory, you can run the command: `sh shell/step3.sh > ./logs/step3.log 2>&1 & disown`.
It will run all step 3 commands showed above.

### Step 4: Post-processing
Post-processing is the focus of this framework. We use the Scikit-Pierre to post-processing the candidate items given by the recommender algorithms provided by the Scikit-Surprise or other recommender library.
The parameters given from the command line are used to create one or more recommendation systems. So, you can change them and create a lot of different systems. It is possible to use the same candidate items set, given as entry to different post-processing formulations.
The recommendations produced by this step are saved in `data/experiment/<dataset_name>/recommendation_lists/`.

#### Run Examples
1. Movielens 20M: `python3 step4_postprocessing.py from_file=YES file_name=ml-20m`    
2. Food.com Recipes: `python3 step4_postprocessing.py from_file=YES file_name=food`     
3. Last.fm: `python3 step4_postprocessing.py from_file=YES file_name=lfm-2b-subset`

In the directory `shell`, there are examples of script to run the **step 4** for the three dataset cited above.
From the main project directory, you can run the command: `sh shell/step4.sh > ./logs/step4.log 2>&1 & disown`.
It will run all step 4 commands showed above.

### Step 5: Metric
In the Metric step, the recommendation lists produced by the post-processing are evaluated.
In version 0.1, the framework provides five metrics: **TIME**, **MAP**, **MRR**, **MACE** and **MRMC**.
In version 0.2, the framework provides news metrics, such as: **ILS**, **COVERAGE**, **UNEXPECTED**, **SERENDIPITY** and **MAMC**.
All metrics are provided by Scikit-Pierre. The Scikit-Surprise provides other metrics, so it is possible to call these too.
The system that you want to evaluate is given by the same parameters from the post-processing.

#### Run Examples
1. Movielens 20M: `python3 step5_metrics.py from_file=YES file_name=ml-20m`    
2. Food.com Recipes: `python3 step5_metrics.py from_file=YES file_name=food`     
3. Last.fm: `python3 step5_metrics.py from_file=YES file_name=lfm-2b-subset`

In the directory `shell`, there are examples of script to run the **step 5** for the three dataset cited above.
From the main project directory, you can run the command: `sh shell/step5.sh > ./logs/step5.log 2>&1 & disown`.
It will run all step 5 commands showed above.

### Step 6: Protocol
Protocol step compiles all metrics results, applying an average along the trials and folds.
The final values are the systems performances in each metric.
Coefficients are applied as a new metric. We inherit the coefficients from Scikit-Pierre.
The final file is saved inside: `results/decision/{dataset}/decision.csv`.

#### Run Examples
1. Movielens 20M: `python3 step6_protocol.py from_file=YES file_name=ml-20m`    
2. Food.com Recipes: `python3 step6_protocol.py from_file=YES file_name=food`     
3. Last.fm: `python3 step6_protocol.py from_file=YES file_name=lfm-2b-subset`

In the directory `shell`, there are examples of script to run the **step 6** for the three dataset cited above.
From the main project directory, you can run the command: `sh shell/step6.sh > ./logs/step6.log 2>&1 & disown`.
It will run all step 6 commands showed above.

### Step 7: Charts
The Chart step uses the compiled raw results from the Protocol step to produce figures, tables, comparisons and research questions.
It is the last step on the framework. The result files are saved inside: `results/graphics/results/{dataset}/{filename}`.

#### Run Examples
1. Movielens 20M: `python3 step7_charts_tables.py from_file=YES file_name=ml-20m`    
2. Food.com Recipes: `python3 step7_charts_tables.py from_file=YES file_name=food`     
3. Last.fm: `python3 step7_charts_tables.py from_file=YES file_name=lfm-2b-subset`