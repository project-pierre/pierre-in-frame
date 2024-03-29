#!/bin/bash
cd ./pierre_in_frame && python3 step5_metrics.py from_file=YES file_name=step5_steck > ../logs/step5_steck.log 2>&1 & disown