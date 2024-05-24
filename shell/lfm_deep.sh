#!/bin/bash
cd ./pierre_in_frame &&
python3 step2_searches.py from_file=YES file_name=deep_ae_lfm-2b-subset &&
python3 step3_processing.py from_file=YES file_name=deep_ae_lfm-2b-subset &&
python3 step4_postprocessing.py from_file=YES file_name=deep_ae_lfm-2b-subset &&
python3 step5_metrics.py from_file=YES file_name=deep_ae_lfm-2b-subset &&
python3 step6_protocol.py from_file=YES file_name=lfm-2b-subset &&
python3 step7_charts_tables.py from_file=YES file_name=lfm-2b-subset
