#!/bin/bash
spm_train --input=data/target.txt --model_prefix=target --vocab_size=8000 --character_coverage=1.0 --model_type=unigram