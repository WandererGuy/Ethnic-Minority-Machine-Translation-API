#!/bin/bash
spm_train --input=data/source.txt --model_prefix=source --vocab_size=8000 --character_coverage=1.0 --model_type=unigram
