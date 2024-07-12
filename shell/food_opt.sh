#!/bin/bash
cd ./pierre_in_frame &&
python3 step1_preprocessing.py from_file=YES file_name=food &&
python3 step2_searches.py from_file=YES file_name=food
