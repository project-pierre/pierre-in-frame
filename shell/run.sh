#!/bin/bash
cd ./pierre_in_frame &&
python3 step2_searches.py from_file=YES file_name=step2 &&
python3 step3_processing.py from_file=YES file_name=step3 &&
python3 step4_postprocessing.py from_file=YES file_name=step4 &&
python3 step5_metrics.py from_file=YES file_name=step5