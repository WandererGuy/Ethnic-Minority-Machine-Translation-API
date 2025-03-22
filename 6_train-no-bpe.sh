#!/bin/bash
# onmt-build-vocab --from_vocab source.vocab --from_format sentencepiece --save_vocab vocab/example.vocab.src
# onmt-build-vocab --from_vocab target.vocab --from_format sentencepiece --save_vocab vocab/example.vocab.tgt
onmt_build_vocab -config khmer-viet-no-bpe.yaml -n_sample 10000
# onmt_train -config khmer-viet-no-bpe.yaml -verbose -train_from models/run2/model_step_136000.pt
onmt_train -config khmer-viet-no-bpe.yaml -verbose --log_file output_train_log/output.log --tensorboard --tensorboard_log_dir output_tensorboard_log
