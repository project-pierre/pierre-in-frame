#!/bin/bash
cd ./pierre_in_frame && python3 step7_charts_tables.py from_file=YES file_name=step7_steck > ../logs/step7_steck.log 2>&1 & disown