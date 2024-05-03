#!/bin/bash
cd ./pierre_in_frame &&

python3 step3_processing.py from_file=YES file_name=lfm-2b-subset &&

python3 step1_preprocessing.py from_file=YES file_name=step1_lfm_distribution &&
python3 step1_preprocessing.py from_file=YES file_name=step1_ohe_lfm &&

python3 step2_searches.py from_file=YES file_name=deep_ae_food &&
python3 step2_searches.py from_file=YES file_name=deep_ae_ml-20m &&
python3 step2_searches.py from_file=YES file_name=deep_ae_lfm-2b-subset &&

python3 step3_processing.py from_file=YES file_name=deep_ae_food &&
python3 step3_processing.py from_file=YES file_name=deep_ae_ml-20m &&
python3 step3_processing.py from_file=YES file_name=deep_ae_lfm-2b-subset