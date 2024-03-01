#!/bin/bash
cd ./pierre_in_frame &&
python3 step3_processing.py from_file=YES file_name=ml-1m &&
python3 step3_processing.py from_file=YES file_name=food &&
python3 step3_processing.py from_file=YES file_name=ml-20m &&
python3 step3_processing.py from_file=YES file_name=lfm-2b-subset