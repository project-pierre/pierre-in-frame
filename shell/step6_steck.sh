#!/bin/bash
cd ../code && python3 step6_protocol.py from_file=YES file_name=step6_steck > ../logs/step6_steck.log 2>&1 & disown