from fastapi import FastAPI, HTTPException, Form, APIRouter
from routers.model import MyHTTPException, \
                        my_exception_handler, \
                        reply_bad_request, \
                        reply_server_error, \
                        reply_success

import os 
import uuid
# import nltk
# from nltk.tokenize.treebank import TreebankWordDetokenizer
# from nltk import word_tokenize
import subprocess
import requests
import configparser
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch 

router = APIRouter()

# detokenizer = TreebankWordDetokenizer()
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
static_folder = os.path.join(parent_dir, "static")
translate_output_folder = os.path.join(static_folder, "translate_output")
os.makedirs(translate_output_folder, exist_ok=True)
ckpt_opennmt_folder = os.path.join(parent_dir, "checkpoints")
os.makedirs(ckpt_opennmt_folder, exist_ok=True)



tokenized_folder = os.path.join(static_folder, "tokenized_file")
os.makedirs(tokenized_folder, exist_ok=True)
tokenized_for_infer_folder = os.path.join(static_folder,"tokenized_for_infer")
os.makedirs(tokenized_for_infer_folder, exist_ok=True)

ckpt_tokenizer_folder = os.path.join(parent_dir, "checkpoints")
os.makedirs(ckpt_tokenizer_folder, exist_ok=True)
split_folder = os.path.join(static_folder,"split_target_source")
os.makedirs(split_folder, exist_ok=True)

refined_file_to_translate_folder = os.path.join(parent_dir, "static", "refined_file_to_translate")
os.makedirs(refined_file_to_translate_folder, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print (device)

mtet_model_name = "VietAI/envit5-translation"
mtet_tokenizer = AutoTokenizer.from_pretrained(mtet_model_name)  
mtet_model = AutoModelForSeq2SeqLM.from_pretrained(mtet_model_name)
mtet_model.cuda()


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

# def detokenize_file(src_path, output_path, file_to_translate):
#     vi_output = []
#     en_output = []
#     with open (file_to_translate, 'r') as f:
#         need_translate_lines = f.readlines()
#     with open(src_path, 'r') as f:
#         lines = f.readlines()
        
#         for line in lines:
#             line = line.strip()
#             line = line.replace(" ", "")
#             line = line.replace("▁", " ")
#             en_output_line = detokenizer.detokenize(word_tokenize(line))
#             en_output.append(en_output_line)
#             # output.append(output_line)
#             vi_output_line = mtet_translate_en(language = "en", inputs = en_output_line)
            
#             vi_output.append(vi_output_line)

#     with open(output_path, 'w') as f:
#         for index, line in enumerate(en_output):
#             f.write(need_translate_lines[index])
#             f.write("\n")

#             if en_output[index] == None:
#                 f.write("")
#             else:
#                 f.write(en_output[index])
#             f.write("\n")

#             if vi_output[index] == None:
#                 f.write("")
#             else:
#                 f.write(vi_output[index])

#             f.write("\n")
#             f.write("____________________________________________________________________________________")
#             f.write("\n")

english_tokenization_checkpoints = "checkpoints/english.model"
def detokenize_file(src_path, output_path, file_to_translate):
    vi_output = []
    en_output = []
    with open (file_to_translate, 'r') as f:
        need_translate_lines = f.readlines()

    import subprocess
    model_path = english_tokenization_checkpoints
    input = src_path
    intermediate_output = src_path.replace(".txt", "detokenized.txt")
    command = f"spm_decode --model={model_path} --input_format=piece < {input} > {intermediate_output}"
    print (command)
    # Running the subprocess with the provided command
    subprocess.run(command, shell=True, check=True)    
    time.sleep(3)
    with open(intermediate_output, 'r') as f:
        lines = f.readlines()
        for line in lines:
            en_output_line = line.strip()
            if not en_output_line.endswith("."):
                en_output_line = en_output_line + "."
            en_output.append(en_output_line)
            # output.append(output_line)
            vi_output_line = mtet_translate_en(language = "en", inputs = en_output_line)
            
            vi_output.append(vi_output_line)

    with open(output_path, 'w') as f:
        for index, line in enumerate(en_output):
            f.write(need_translate_lines[index])
            f.write("\n")

            if en_output[index] == None:
                f.write("")
            else:
                f.write(en_output[index])
            f.write("\n")

            if vi_output[index] == None:
                f.write("")
            else:
                f.write(vi_output[index])

            f.write("\n")
            f.write("____________________________________________________________________________________")
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
    
tokenized_for_infer_folder = os.path.join(static_folder,"tokenized_for_infer")

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

def send_request_tokenize(file_to_translate: str, 
                          source_checkpoint_tokenizer_path: str 
                          ):
    # host_ip, port_num = load_main_tokenize_config()
    # url_tokenize = f"http://{host_ip}:{port_num}/tokenize-file"
    
    # payload = {"file_to_tokenize_path" : file_to_translate, 
    #            "source_checkpoint_tokenizer_path" : source_checkpoint_tokenizer_path
    # }
    # files=[
    # ]
    # headers = {}
    # response = requests.request("POST", url_tokenize, headers=headers, data=payload, files=files)
    # return response
    file_to_tokenize_path = file_to_translate
    if not os.path.exists(file_to_tokenize_path):
        raise MyHTTPException(status_code=404, message = f"{file_to_tokenize_path} not found")
    if not os.path.exists(source_checkpoint_tokenizer_path):
        raise MyHTTPException(status_code=404, message = f"{source_checkpoint_tokenizer_path} not found")
    tokenized_for_infer_file = tokenize_file_infer(
                  filepath = file_to_tokenize_path, 
                  source_checkpoint_tokenizer_path=source_checkpoint_tokenizer_path)    
    return tokenized_for_infer_file


import time 
def running_python(command):
    print ("************************************")
    print (" ".join(command))
    # Running the subprocess with the provided command
    subprocess.run(command, text=True)
    time.sleep(4)


def refine_file_to_translate_func(file_path):
    refined_file_to_translate = os.path.join(refined_file_to_translate_folder, str(uuid.uuid4()) + ".txt")
    output = []
    with open(file_path, "r", errors="ignore") as f:
        lines = f.readlines()
        for index, line in enumerate(lines): 
            line = line.strip()
            if line == "":
                continue
            output.append(line)
    with open(refined_file_to_translate, "w", errors="ignore") as f:
        for index, line in enumerate(output):
            if index == len(output) - 1:
                f.write(line)
            else:
                f.write(line)
                f.write("\n")
    return refined_file_to_translate

def mtet_translate_en(
    language: str,
    inputs: str
):
    # new = []
    # with open(input_text_file, "r") as f:
    #     input_text_lines = f.readlines()
    #     for line in input_text_lines:
    #         line = line.strip()
            
    #         if not line.endswith("."):
    #             line += "."
    #         new.append(line)
    # input_text = " ".join(input_text_lines)
    # inputs = input_text
    if language == "en":
        text = list("en: " + inputs)
    else: 
        return inputs
    outputs = mtet_model.generate(mtet_tokenizer(inputs, return_tensors="pt", padding=True).input_ids.to('cuda'), max_length=512)
    text = mtet_tokenizer.batch_decode(outputs, skip_special_tokens=True)
    res = text[0].replace("vi: ", "")
    return res

def calculate_num_lines(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()
        num_lines = len(lines)
        return num_lines


import shutil
@router.post("/train-opennmt")
async def train_opennmt(
    target_source_file: str = Form(...),
    checkpoint_name_prefix: str = Form(...)
    ):
    if not os.path.exists(target_source_file):
        raise MyHTTPException(status_code=404, message = f"{target_source_file} not found")
    data_folder = os.path.join(parent_dir, "data")
    os.makedirs(data_folder, exist_ok=True)
    shutil.copy(target_source_file, data_folder)
    os.rename(os.path.join(data_folder, target_source_file), os.path.join(data_folder, "target_source.txt"))
    time.sleep(5)
    command = ["python", "START_train.py"]
    running_python(command)
    time.sleep(5)
    shutil.copy("models/run2/model_step_130000.pt", os.path.join(parent_dir, "checkpoints"))
    os.rename (os.path.join(parent_dir, "checkpoints", "model_step_130000.pt"), os.path.join(parent_dir, "checkpoints", f"{checkpoint_name_prefix}_model_step_130000.pt"))
    return reply_success(message = "Done and saved new checkpoint", result=None)

@router.get("/get-checkpoint-opennmt-path")
async def get_checkpoint_opennmt_path():
    filenames = os.listdir(ckpt_opennmt_folder)
    filenames = [os.path.join(ckpt_opennmt_folder, name) for name in filenames if not name.endswith(".model")]
    return reply_success(message = "Done", result=filenames)


@router.post("/translate-language/")
async def translate_language(
    # file_to_translate: str = Form(...),
    # lang_source: str = Form(...),
    file_to_translate_content: str = Form(...),
    model_checkpoint_path : str = Form(...),
    source_checkpoint_tokenizer_path: str = Form(...)
):
    # file_contents = await file_to_translate.read()
    # content_str = file_contents.decode('utf-8')  # Assuming the file is UTF-8 encoded
    file_to_translate_folder = os.path.join(static_folder, "file_to_translate")
    os.makedirs(file_to_translate_folder, exist_ok=True)
    new_save_path = os.path.join(file_to_translate_folder, str(uuid.uuid4()) + ".txt")  # Adjust the path as needed
    with open(new_save_path, "w", encoding='utf-8') as new_file:
        new_file.write(file_to_translate_content)
    file_to_translate = new_save_path
    refined_file_to_translate = refine_file_to_translate_func(file_to_translate)
    print ("****** original file after refined ******")
    print (refined_file_to_translate)
    checkpoint_folder = "checkpoints"
    os.makedirs(checkpoint_folder, exist_ok=True)
    host_ip, port_num = load_main_config()

    # allow_lang = ["khmer", "ede"]
    # if lang_source not in allow_lang:
    #     raise MyHTTPException(status_code=400, message="Language not supported")

    res = send_request_tokenize(file_to_translate = refined_file_to_translate, 
                            source_checkpoint_tokenizer_path = source_checkpoint_tokenizer_path
                            )
    # res_json = handle_response(res)
    # tokenized_file = res_json["result"]
    tokenized_file = res
    output_filepath = os.path.join(translate_output_folder, str(uuid.uuid4()) + ".txt")

    if calculate_num_lines(refined_file_to_translate) != calculate_num_lines(tokenized_file):
        raise MyHTTPException(status_code=500, message="Tokenization failed with wrong number of lines")
    
    translate_file(
        model_path=model_checkpoint_path,
        src_path=tokenized_file,
        output_path=output_filepath)
    detokenized_output_filepath = output_filepath.replace(".txt", "_detokenized.txt")

    if calculate_num_lines(refined_file_to_translate) != calculate_num_lines(output_filepath):
        raise MyHTTPException(status_code=500, message="TRANSLATION failed with wrong number of lines")

    detokenize_file(src_path=output_filepath, 
                    output_path=detokenized_output_filepath, 
                    file_to_translate = refined_file_to_translate
                    )
    t = rf"{os.path.basename(translate_output_folder)}/{os.path.basename(detokenized_output_filepath)}"
    
    # t_path = os.path.join(static_folder, os.path.basename(translate_output_folder), os.path.basename(detokenized_output_filepath))
    # with open(t_path, "r") as file:
    #     content = file.read()

    url = f"http://127.0.0.1:{port_num}/static/{t}"
    return reply_success(
        message="Translation success",
        result=url)





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
    filenames = [os.path.join(ckpt_tokenizer_folder, name) for name in filenames if name.endswith(".model")]
    return reply_success(message = "Done", result=filenames)



