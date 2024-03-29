#!/bin/bash
cd ./pierre_in_frame &&
python3 step4_postprocessing.py from_file=YES file_name=food_steck &&
python3 step4_postprocessing.py from_file=YES file_name=ml-20m_steck &&
python3 step4_postprocessing.py from_file=YES file_name=lfm-2b-subset_steck