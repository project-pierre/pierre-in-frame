#!/bin/bash
python3 step1_preprocessing.py from_file=YES file_name=step1_ml-20m_distribution &&
python3 step1_preprocessing.py from_file=YES file_name=step1_ohe_ml-20m &&
python3 step2_searches.py from_file=YES file_name=ml-20m &&
python3 step3_processing.py from_file=YES file_name=ml-20m &&
python3 step4_postprocessing.py from_file=YES file_name=ml-20m_steck &&
python3 step5_metrics.py from_file=YES file_name=ml-20m_steck
