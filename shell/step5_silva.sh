#!/bin/bash
cd ./code && python3 step5_metrics.py from_file=YES file_name=step5_silva > ../logs/step5_silva.log 2>&1 & disown