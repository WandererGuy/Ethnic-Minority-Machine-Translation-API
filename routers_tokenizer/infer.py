from fastapi import FastAPI, HTTPException, Form, APIRouter
from routers.model import MyHTTPException, \
                        my_exception_handler, \
                        reply_bad_request, \
                        reply_server_error, \
                        reply_success

import os 
import uuid
import subprocess

router = APIRouter()
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
static_folder = os.path.join(parent_dir, "static")
tokenized_folder = os.path.join(static_folder, "tokenized_file")
os.makedirs(tokenized_folder, exist_ok=True)
tokenized_for_infer_folder = os.path.join(static_folder,"tokenized_for_infer")
os.makedirs(tokenized_for_infer_folder, exist_ok=True)

ckpt_tokenizer_folder = os.path.join(parent_dir, "checkpoint_tokenizer")
os.makedirs(ckpt_tokenizer_folder, exist_ok=True)
split_folder = os.path.join(static_folder,"split_target_source")
os.makedirs(split_folder, exist_ok=True)



def split_target_source(target_source_file, output_src_file, output_target_file):
    path = target_source_file
    with open (path, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip("\n")
            tab_count = line.count('\t')
            if tab_count != 1:
                raise Exception("each line must have 1 tab only between source and target sentence")


    path = target_source_file
    with open (path, "r") as f:
        lines = f.readlines()
        source_single_line_ls = []
        target_single_line_ls = []
        for line in lines:
            line = line.strip("\n")
            target, source = line.split("\t")
            source_single_line_ls.append(source)
            target_single_line_ls.append(target)
    source_dest_path = output_src_file
    with open (source_dest_path, "w") as f:
        for line in source_single_line_ls:
            f.write(line)
            f.write("\n")
    target_dest_path = output_target_file
    with open (target_dest_path, "w") as f:
        for line in target_single_line_ls:
            f.write(line)
            f.write("\n")


def tokenize_file_infer(filepath, source_checkpoint_tokenizer_path):
    tokenized_for_infer_file = os.path.join(tokenized_for_infer_folder , str(uuid.uuid4()) + ".txt")
    command = ["spm_encode",
                f"--model={source_checkpoint_tokenizer_path}",
                "--input", filepath, 
                "--output", tokenized_for_infer_file]
    print (" ".join(command))
    # Running the subprocess with the provided command
    result = subprocess.run(command, capture_output=True, text=True)
    return tokenized_for_infer_file


import shutil
def train_spm(train_file, vocab_size):
    command = f"spm_train --input={train_file} --model_prefix=source --vocab_size={vocab_size} --character_coverage=1.0 --model_type=unigram"
    try:
        subprocess.run(command, shell=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        return False
    
def train_init(train_file, train_tokenizer_name):
    import time 
    for vocab_size in [8000, 4000, 2000]:
        if train_spm(train_file, vocab_size) == False:
            print ("******************** reduce vocab size ********************")
            time.sleep(5)
            continue 
        else:
            break
    os.rename("source.model", train_tokenizer_name + ".model")
    save_path = os.path.join(ckpt_tokenizer_folder, train_tokenizer_name + ".model")
    if os.path.exists(save_path):
        os.remove(save_path)
    time.sleep(3)
    shutil.move(train_tokenizer_name + ".model", ckpt_tokenizer_folder)
    return save_path

@router.post("/split-target-source-file")
async def split_target_source_file(
    target_source_file: str = Form(...)
    ):
    if not os.path.exists(target_source_file):
        raise MyHTTPException(status_code=404, message = f"{target_source_file} not found")
    output_src_file = os.path.join(split_folder, str(uuid.uuid4()) + ".txt")
    output_target_file = os.path.join(split_folder, str(uuid.uuid4()) + ".txt")
    split_target_source(target_source_file, output_src_file, output_target_file)
    return reply_success(message = "Done", result={"source": output_src_file, "target": output_target_file})


@router.post("/train-tokenizer")
async def train_tokenizer(
    train_file: str = Form(...),
    train_tokenizer_name: str = Form(...)
    ):
    if not os.path.exists(train_file):
        raise MyHTTPException(status_code=404, message = f"{train_file} not found")
    save_path = train_init(train_file, train_tokenizer_name)
    return reply_success(message = "Done, model saved in ", result=save_path)

@router.get("/get-checkpoint-tokenizer-path")
async def get_checkpoint_tokenizer_path():
    filenames = os.listdir(ckpt_tokenizer_folder)
    filenames = [os.path.join(ckpt_tokenizer_folder, name) for name in filenames]
    return reply_success(message = "Done", result=filenames)


@router.post("/tokenize-file")
async def tokenize_file(
    file_to_tokenize_path: str = Form(...),
    source_checkpoint_tokenizer_path: str = Form(...)
    ):
    if not os.path.exists(file_to_tokenize_path):
        raise MyHTTPException(status_code=404, message = f"{file_to_tokenize_path} not found")
    if not os.path.exists(source_checkpoint_tokenizer_path):
        raise MyHTTPException(status_code=404, message = f"{source_checkpoint_tokenizer_path} not found")
    tokenized_for_infer_file = tokenize_file_infer(
                  filepath = file_to_tokenize_path, 
                  source_checkpoint_tokenizer_path=source_checkpoint_tokenizer_path)    
    return reply_success(message = "Done", result=tokenized_for_infer_file)