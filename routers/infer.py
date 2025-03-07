from fastapi import FastAPI, HTTPException, Form, APIRouter
from routers.model import MyHTTPException, \
                        my_exception_handler, \
                        reply_bad_request, \
                        reply_server_error, \
                        reply_success

import os 
import uuid
import nltk
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk import word_tokenize
import subprocess
import requests
import configparser

router = APIRouter()

detokenizer = TreebankWordDetokenizer()
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
static_folder = os.path.join(parent_dir, "static")
translate_output_folder = os.path.join(static_folder, "translate_output")
os.makedirs(translate_output_folder, exist_ok=True)
ckpt_opennmt_folder = os.path.join(parent_dir, "checkpoint_OpenNMT")
os.makedirs(ckpt_opennmt_folder, exist_ok=True)
random_folder = os.path.join(parent_dir, "checkpoint_tokenizer")
os.makedirs(random_folder, exist_ok=True)



def load_main_config():
    config = configparser.ConfigParser()
    config.read(os.path.join(parent_dir,'config','config.ini'))
    host_ip = config['DEFAULT']['host'] 
    port_num = config['DEFAULT']['port'] 
    return host_ip, port_num
def load_main_tokenize_config():
    config = configparser.ConfigParser()
    config.read(os.path.join(parent_dir,'config','config_tokenize.ini'))
    host_ip = config['DEFAULT']['host'] 
    port_num = config['DEFAULT']['port'] 
    return host_ip, port_num

# def select_checkpoint(lang_source):
#     checkpoint_folder = "language_checkpoints"
#     if lang_source == "khmer":
#         checkpoint_path = os.path.join(checkpoint_folder, "khmer_model_step_125000.pt")
#     return checkpoint_path


def translate_file(model_path, src_path, output_path):
    command = ["onmt_translate",
            "--model", model_path,
            "--src", src_path,
            "--output", output_path,
            "--verbose",
            "--gpu", "0"]
    print ("start process ....")
    print ("running command", " ".join(command))
    process = subprocess.run(command,
            capture_output=True,
            shell=False, 
            text=True,
            encoding = 'utf-8')
    # Checking if the process was successful

    if process.returncode == 0:
        # Process stdout (translation output)
        print(process.stdout)
    else:
        # If there was an error, print stderr
        print("Error:", process.stderr)
    print ("end process ....")


def detokenize_file(src_path, output_path):
    output = []
    with open(src_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            line = line.replace(" ", "")
            line = line.replace("▁", " ")
            output.append(detokenizer.detokenize(word_tokenize(line)))

    with open(output_path, 'w') as f:
        for line in output:
            f.write(line)
            f.write("\n")


def handle_response(response):
    # Check if the response is in JSON format
    if response.status_code == 200:  # Check if the request was successful
        try:
            json_data = response.json()  # Parse the JSON response
            return json_data  # Print the parsed JSON
        except ValueError:
            raise MyHTTPException(status_code=500, message = "send Request to another Backend, have a Response is not valid JSON")
    else:
        raise MyHTTPException(status_code=500, message = f"send Request to another Backend, failed with status code {response.status_code}")
    
def send_request_tokenize(file_to_translate: str, 
                          train_tokenizer: str 
                          ):
    host_ip, port_num = load_main_tokenize_config()
    url_tokenize = f"http://{host_ip}:{port_num}/tokenize-file"

    payload = {"file_to_tokenize_path" : file_to_translate, 
               "train_tokenizer" : train_tokenizer
    }
    files=[
    ]
    headers = {}
    response = requests.request("POST", url_tokenize, headers=headers, data=payload, files=files)
    return response
import time 
def running_python(command):
    print ("************************************")
    print (" ".join(command))
    # Running the subprocess with the provided command
    subprocess.run(command, text=True)
    time.sleep(4)

import shutil
@router.post("/train-opennmt")
async def train_opennmt(
    target_source_file: str = Form(...)
    ):
    data_folder = os.path.join(parent_dir, "data")
    shutil.copy(target_source_file, data_folder)
    os.rename(os.path.join(data_folder, target_source_file), os.path.join(data_folder, "target_source.txt"))
    time.sleep(5)
    command = ["python", "START_train.py"]
    running_python(command)
    time.sleep(5)
    shutil.copy("models/run2/model_step_130000.pt", os.path.join(parent_dir, "checkpoint_OpenNMT"))
    return reply_success(message = "Done", result=None)

@router.get("/get-checkpoint-opennmt-path")
async def get_checkpoint_opennmt_path():
    filenames = os.listdir(ckpt_opennmt_folder)
    filenames = [os.path.join(ckpt_opennmt_folder, name) for name in filenames]
    return reply_success(message = "Done", result=filenames)


@router.post("/translate-language/")
async def translate_language(
    file_to_translate: str = Form(...),
    # lang_source: str = Form(...),
    model_checkpoint_path : str = Form(...),
    source_train_tokenizer: str = Form(...)
):
    checkpoint_folder = "checkpoint_OpenNMT"
    os.makedirs(checkpoint_folder, exist_ok=True)
    host_ip, port_num = load_main_config()

    # allow_lang = ["khmer", "ede"]
    # if lang_source not in allow_lang:
    #     raise MyHTTPException(status_code=400, message="Language not supported")

    res = send_request_tokenize(file_to_translate = file_to_translate, 
                            train_tokenizer = source_train_tokenizer
                            )
    res_json = handle_response(res)
    tokenized_file = res_json["result"]
    output_filepath = os.path.join(translate_output_folder, str(uuid.uuid4()) + ".txt")

    translate_file(
        model_path=model_checkpoint_path,
        src_path=tokenized_file,
        output_path=output_filepath)
    detokenized_output_filepath = output_filepath.replace(".txt", "_detokenized.txt")
    detokenize_file(src_path=output_filepath, 
                    output_path=detokenized_output_filepath)
    url = f"http://{host_ip}:{port_num}/static/{os.path.basename(translate_output_folder)}/{os.path.basename(detokenized_output_filepath)}"
    return reply_success(
        message="Translation success",
        result=url)


