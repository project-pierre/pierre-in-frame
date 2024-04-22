#!/bin/bash
cd ./pierre_in_frame &&
python3 step1_preprocessing.py from_file=YES file_name=lfm-2b-subset &&
python3 step1_preprocessing.py from_file=YES file_name=step1_lfm_distribution &&
python3 step1_preprocessing.py from_file=YES file_name=step1_ohe_lfm
