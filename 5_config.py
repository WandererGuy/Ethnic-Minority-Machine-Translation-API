# from datetime import datetime

# # Get current date and time
# current_time = datetime.now()

# # Format the timestamp to include year, month, date, hour, and minute
# timestamp = current_time.strftime("%Y-%m-%d-%H-%M")

# def write_train_no_bpe_config():
#     config = f"""# data-no-bpe.yaml

#     ## Where the samples will be written
#     save_data: models/{timestamp}/example
#     ## Where the vocab(s) will be written
#     src_vocab: vocab/example.vocab.src
#     tgt_vocab: vocab/example.vocab.tgt
#     # Prevent overwriting existing files in the folder
#     overwrite: True

#     # Corpus opts:
#     data:
#         corpus_1:
#             path_src: data/src-train-token.txt
#             path_tgt: data/tgt-train-token.txt
#         valid:
#             path_src: data/src-val-token.txt
#             path_tgt: data/tgt-val-token.txt


#     # Train on a single GPU
#     world_size: 1
#     gpu_ranks: [0]

#     # Where to save the checkpoints
#     save_model: models/{timestamp}/model
#     save_checkpoint_steps: 10000
#     train_steps: 130000
#     valid_steps: 1000

#     # Model
#     position_encoding: 'true'
#     enc_layers: 6
#     dec_layers: 6
#     decoder_type: transformer
#     encoder_type: transformer
#     word_vec_size: 512
#     rnn_size: 512
#     layers: 6
#     transformer_ff: 2048
#     heads: 8

#     # Batching
#     queue_size: 10000
#     batch_size: 4096
#     valid_batch_size: 4096
#     batch_type: tokens
#     """

#     with open("khmer-viet-no-bpe.yaml", "w") as f:
#         f.write(config)


# def write_train_bpe_config():
#     config = f"""# khmer-viet.yaml

#     ## Where the samples will be written
#     save_data: models/{timestamp}/example
#     ## Where the vocab(s) will be written
#     src_vocab: example.vocab.src
#     tgt_vocab: example.vocab.tgt
#     # Prevent overwriting existing files in the folder
#     overwrite: True

#     # Corpus opts:
#     data:
#         corpus_1:
#             path_src: data/src-train-bpe.txt
#             path_tgt: data/tgt-train-bpe.txt
#         valid:
#             path_src: data/src-val-bpe.txt
#             path_tgt: data/tgt-val-bpe.txt


#     # Train on a single GPU
#     world_size: 1
#     gpu_ranks: [0]

#     # Where to save the checkpoints
#     save_model: models/{timestamp}/model
#     save_checkpoint_steps: 10000
#     train_steps: 130000
#     valid_steps: 1000

#     # Model
#     position_encoding: 'true'
#     enc_layers: 6
#     dec_layers: 6
#     decoder_type: transformer
#     encoder_type: transformer
#     word_vec_size: 512
#     rnn_size: 512
#     layers: 6
#     transformer_ff: 2048
#     heads: 8

#     # Batching
#     queue_size: 10000
#     batch_size: 4096
#     valid_batch_size: 4096
#     batch_type: tokens
#     """
#     with open("khmer-viet.yaml", "w") as f:
#         f.write(config)

# if __name__ == "__main__":
#     write_train_no_bpe_config()
#     write_train_bpe_config()
#     import os   
#     import time 
#     if not os.path.exists('output_log'):
#         os.makedirs('output_log', exist_ok=True)

from datetime import datetime

# Get current date and time
current_time = datetime.now()

# Format the timestamp to include year, month, date, hour, and minute
timestamp = current_time.strftime("%Y-%m-%d-%H-%M")

model_path = "models/run2/model"
save_data_path = "models/run2/example"
delete_path = "models/run2"
import shutil
import os 
if os.path.exists(delete_path):
    shutil.rmtree(delete_path)
    os.makedirs(delete_path, exist_ok=True)
# https://opennmt.net/OpenNMT-py/FAQ.html#how-do-i-train-the-transformer-model
def write_train_no_bpe_config():
    config = f"""# data-no-bpe.yaml

    ## Where the samples will be written
    save_data: {save_data_path}
    ## Where the vocab(s) will be written
    src_vocab: vocab/example.vocab.src
    tgt_vocab: vocab/example.vocab.tgt
    # Prevent overwriting existing files in the folder
    overwrite: True

    # Corpus opts:
    data:
        corpus_1:
            path_src: data/src-train-token.txt
            path_tgt: data/tgt-train-token.txt
        valid:
            path_src: data/src-val-token.txt
            path_tgt: data/tgt-val-token.txt


    # Where to save the checkpoints
    save_model: {model_path}
    save_checkpoint_steps: 10000
    valid_steps: 10000
    train_steps: 80000

    # Batching
    bucket_size: 262144

    
    num_workers: 4
    batch_type: "tokens"
    batch_size: 4096
    valid_batch_size: 2048
    accum_count: [4]
    accum_steps: [0]

    # Optimization
    model_dtype: "fp16"
    optim: "adam"
    learning_rate: 2
    warmup_steps: 8000
    decay_method: "noam"
    adam_beta2: 0.998
    max_grad_norm: 0
    label_smoothing: 0.1
    param_init: 0
    param_init_glorot: true
    normalization: "tokens"

    
    # Model
    encoder_type: transformer
    decoder_type: transformer
    position_encoding: true
    enc_layers: 6
    dec_layers: 6
    heads: 8
    rnn_size: 512
    word_vec_size: 512
    transformer_ff: 2048
    dropout_steps: [0]
    dropout: [0.1]
    attention_dropout: [0.1]


    world_size: 1
    gpu_ranks:
    - 0         
    """

    with open("khmer-viet-no-bpe.yaml", "w") as f:
        f.write(config)


def write_train_bpe_config():
    config = f"""# khmer-viet.yaml

    ## Where the samples will be written
    save_data: {save_data_path}
    ## Where the vocab(s) will be written
    src_vocab: example.vocab.src
    tgt_vocab: example.vocab.tgt
    # Prevent overwriting existing files in the folder
    overwrite: True

    # Corpus opts:
    data:
        corpus_1:
            path_src: data/src-train-bpe.txt
            path_tgt: data/tgt-train-bpe.txt
        valid:
            path_src: data/src-val-bpe.txt
            path_tgt: data/tgt-val-bpe.txt




    # Where to save the checkpoints
    save_model: {model_path}
    save_checkpoint_steps: 10000
    valid_steps: 10000
    train_steps: 80000

    # Batching
    bucket_size: 262144
    
    
    num_workers: 4
    batch_type: "tokens"
    batch_size: 4096
    valid_batch_size: 2048
    accum_count: [4]
    accum_steps: [0]

    # Optimization
    model_dtype: "fp16"
    optim: "adam"
    learning_rate: 2
    warmup_steps: 8000
    decay_method: "noam"
    adam_beta2: 0.998
    max_grad_norm: 0
    label_smoothing: 0.1
    param_init: 0
    param_init_glorot: true
    normalization: "tokens"

    # Model
    encoder_type: transformer
    decoder_type: transformer
    position_encoding: true
    enc_layers: 6
    dec_layers: 6
    heads: 8
    rnn_size: 512
    word_vec_size: 512
    transformer_ff: 2048
    dropout_steps: [0]
    dropout: [0.1]
    attention_dropout: [0.1]


    world_size: 1
    gpu_ranks:
    - 0                                                             
    """
    with open("khmer-viet.yaml", "w") as f:
        f.write(config)

if __name__ == "__main__":
    write_train_no_bpe_config()
    write_train_bpe_config()
    import os   
    import time 
    if not os.path.exists('output_log'):
        os.makedirs('output_log', exist_ok=True)

