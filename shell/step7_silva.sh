#!/bin/bash
cd ./code && python3 step7_charts_tables.py from_file=YES file_name=step7_silva > ../logs/step7_silva.log 2>&1 & disown