#!/bin/bash
cd ./pierre_in_frame &&
python3 step4_postprocessing.py from_file=YES file_name=step4 &&
python3 step5_metrics.py from_file=YES file_name=step5
