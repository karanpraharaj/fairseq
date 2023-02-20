# Rename files from input to input0. 
# This is because the XMOD trainer expects the input files to be named input0.

import os
import shutil

import os

languages = ['de', 'en', 'es', 'fr', 'zh', 'jp', 'IPT']
categories = ['apparel', 'home', 'musical_instruments', 'sports']

for language in languages:
    if language == 'IPT':
        for lang_ipt in languages:
            path = f"marc/fairseq/IPT/{lang_ipt}/bin/input"
            new_path = f"marc/fairseq/IPT/{lang_ipt}/bin/input0"
            # Check if 'path' exists
            if os.path.exists(path):
                os.rename(path, new_path)
    else:
        for category in categories:
            path = f"marc/fairseq/{language}/{category}/bin/input"
            new_path = f"marc/fairseq/{language}/{category}/bin/input0"
            # Check if 'path' exists
            if os.path.exists(path):
                os.rename(path, new_path)