#! /usr/bin/bash

# may need to specify full path to activate
source activate impact

jupyter nbconvert --config nbcconf.py >out_sloth.txt 2>&1
