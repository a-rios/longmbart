#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from pathlib import Path

de_files = Path('/srv/scratch6/kew/mbart/longmbart/dummy/de/raw')
en_files = Path('/srv/scratch6/kew/mbart/longmbart/dummy/en/raw')


for de_file, en_file in zip(sorted(de_files.iterdir()), sorted(en_files.iterdir())):
    print(de_file, en_file)

