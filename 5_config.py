from datetime import datetime

# Get current date and time
current_time = datetime.now()

# Format the timestamp to include year, month, date, hour, and minute
# timestamp = current_time.strftime("%Y-%m-%d-%H-%M")
timestamp = "run2"
def write_train_no_bpe_config():
    config = f"""# data-no-bpe.yaml

    ## Where the samples will be written
    save_data: data/{timestamp}/example
    ## Where the vocab(s) will be written
    src_vocab: data/{timestamp}/example.vocab.src
    tgt_vocab: data/{timestamp}/example.vocab.tgt
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

    # Vocabulary files that were just created
    src_vocab: models/{timestamp}/example.vocab.src
    tgt_vocab: models/{timestamp}/example.vocab.tgt

    # Train on a single GPU
    world_size: 1
    gpu_ranks: [0]

    # Where to save the checkpoints
    save_model: models/{timestamp}/model
    save_checkpoint_steps: 10000
    train_steps: 130000
    valid_steps: 1000

    # Model
    position_encoding: 'true'
    enc_layers: 6
    dec_layers: 6
    decoder_type: transformer
    encoder_type: transformer
    word_vec_size: 512
    rnn_size: 512
    layers: 6
    transformer_ff: 2048
    heads: 8

    # Batching
    queue_size: 10000
    batch_size: 4096
    valid_batch_size: 4096
    batch_type: tokens
    """

    with open("khmer-viet-no-bpe.yaml", "w") as f:
        f.write(config)


def write_train_bpe_config():
    config = f"""# khmer-viet.yaml

    ## Where the samples will be written
    save_data: data/{timestamp}/example
    ## Where the vocab(s) will be written
    src_vocab: data/{timestamp}/example.vocab.src
    tgt_vocab: data/{timestamp}/example.vocab.tgt
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

    # Vocabulary files that were just created
    src_vocab: models/{timestamp}/example.vocab.src
    tgt_vocab: models/{timestamp}/example.vocab.tgt

    # Train on a single GPU
    world_size: 1
    gpu_ranks: [0]

    # Where to save the checkpoints
    save_model: models/{timestamp}/model
    save_checkpoint_steps: 10000
    train_steps: 130000
    valid_steps: 1000

    # Model
    position_encoding: 'true'
    enc_layers: 6
    dec_layers: 6
    decoder_type: transformer
    encoder_type: transformer
    word_vec_size: 512
    rnn_size: 512
    layers: 6
    transformer_ff: 2048
    heads: 8

    # Batching
    queue_size: 10000
    batch_size: 4096
    valid_batch_size: 4096
    batch_type: tokens
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

