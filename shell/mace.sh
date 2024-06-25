#!/bin/bash
cd ./pierre_in_frame &&
python3 step5_metrics.py from_file=YES file_name=mace &&
python3 step6_protocol.py from_file=YES file_name=step6 &&
python3 step7_charts_tables.py from_file=YES file_name=step7