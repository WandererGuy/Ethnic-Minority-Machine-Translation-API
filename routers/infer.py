from fastapi import FastAPI, HTTPException, Form, APIRouter
from routers.model import MyHTTPException, \
                        my_exception_handler, \
                        reply_bad_request, \
                        reply_server_error, \
                        reply_success

import os 
import uuid
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

def select_checkpoint(lang_source):
    checkpoint_folder = "language_checkpoints"
    if lang_source == "khmer":
        checkpoint_path = os.path.join(checkpoint_folder, "khmer_model_step_125000.pt")
    return checkpoint_path


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
    
def send_request_tokenize(file_to_tokenize_path: str, 
                          lang_source: str
                          ):
    host_ip, port_num = load_main_tokenize_config()
    url_tokenize = f"http://{host_ip}:{port_num}/tokenize-language"

    payload = {"file_to_tokenize_path" : file_to_tokenize_path, 
               "lang_source" : lang_source
    }
    files=[
    ]
    headers = {}
    response = requests.request("POST", url_tokenize, headers=headers, data=payload, files=files)
    return response

@router.post("/translate-language/")
async def translate_language(
    file_to_translate_path: str = Form(...),
    lang_source: str = Form(...),
):
    host_ip, port_num = load_main_config()

    allow_lang = ["khmer"]
    if lang_source not in allow_lang:
        raise MyHTTPException(status_code=400, message="Language not supported")

    res = send_request_tokenize(file_to_tokenize_path = file_to_translate_path, 
                            lang_source = lang_source
                            )
    res_json = handle_response(res)
    tokenized_file_path = res_json["result"]
    output_filepath = os.path.join(translate_output_folder, str(uuid.uuid4()) + ".txt")

    translate_file(
        model_path=select_checkpoint(lang_source),
        src_path=tokenized_file_path,
        output_path=output_filepath)

    detokenize_file(src_path=output_filepath, 
                    output_path=output_filepath)
    url = f"http://{host_ip}:{port_num}/static/{os.path.basename(translate_output_folder)}/{os.path.basename(output_filepath)}"
    return reply_success(
        message="Translation success",
        result=url)