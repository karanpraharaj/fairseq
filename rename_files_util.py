# Rename files from input to input0. 
# This is because the XMOD trainer expects the input files to be named input0.

import os
import shutil

import os

languages = ['de', 'en', 'es', 'fr', 'hi', 'th']

for language in languages:
    for i in range(5):
        path = f"mtop_partitioned/fairseq/{language}/{i}/bin/input"
        new_path = f"mtop_partitioned/fairseq/{language}/{i}/bin/input0"
        # Check if 'path' exists
        if os.path.exists(path):
            os.rename(path, new_path)