#!/bin/bash
python asyncsgd.py --task_index=0 &
sleep 2 # wait for variable to be initialized
python asyncsgd.py --task_index=1 &
python asyncsgd.py --task_index=2 &
python asyncsgd.py --task_index=3 &
python asyncsgd.py --task_index=4 &

