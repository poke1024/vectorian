#!/bin/bash
cd "$(dirname "$0")"
pip install pdoc3 holoviews
export VECTORIAN_CPP_IMPORT=1
export PYTHONPATH=$PYTHONPATH:"$(dirname "$0")/../"
pdoc --html --output-dir build -c latex_math=True --force "$(dirname "$0")/../vectorian"
