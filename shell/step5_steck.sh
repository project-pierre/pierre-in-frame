#!/bin/bash
cd ./pierre_in_frame &&
python3 step5_metrics.py from_file=YES file_name=lfm-2b-subset &&
python3 step5_metrics.py from_file=YES file_name=lfm-2b-subset_p