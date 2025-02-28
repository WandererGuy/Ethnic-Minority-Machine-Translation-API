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
tokenized_for_infer_folder = "tokenized_for_infer"
os.makedirs(tokenized_for_infer_folder, exist_ok=True)

def tokenize_file(filepath, lang_source, trained_tokenizer_name):
    # if lang_source == "khmer":
    #     from khmernltk import word_tokenize
    #     def tokenize_line(line):
    #         '''
    #         args: line (raw text)
    #         return: new_res (tokenized text)
    #         progress:
    #         - tokenize/subword into a list of tokens/subwords
    #         - make sure each token separate by space (openNMT input standard)
    #         - and token contains space in between , replace space with _
    #         '''
    #         line = line.strip()
    #         res = word_tokenize(line, return_tokens=True)
    #         new_res = []
    #         for token in res:
    #             if token.strip() == '': continue
    #             new_token = token.replace(' ', '_')
    #             new_res.append(new_token)
    #         new_res = " ".join(new_res)
    #         return new_res
    #     tokenized_file_name = str(uuid.uuid4()) + ".txt"
    #     tokenized_file_path = os.path.join(tokenized_folder , tokenized_file_name)
    #     f_token = open(tokenized_file_path, mode='w', encoding='utf8')
    #     f_raw = open(filepath, mode='r', encoding='utf8')
    #     n = 0
    #     for line in f_raw:
    #         f_token.write(tokenize_line(line) + "\n")
    #         n+=1
    #     print('Số lượng câu trong tập huấn luyện nguồn là ', n)
    #     return tokenized_file_path
    tokenized_for_infer_file = os.path.join(tokenized_for_infer_folder , str(uuid.uuid4()) + ".txt")
    command = ["spm_encode",
                f"--model={trained_tokenizer_name}.model",
                "--input", filepath, 
                "--output", tokenized_for_infer_file]
    print (" ".join(command))
    # Running the subprocess with the provided command
    result = subprocess.run(command, capture_output=True, text=True)
    return tokenized_for_infer_file



@router.post("/tokenize-language/")
async def tokenize_language(
    file_to_tokenize_path: str = Form(...),
    lang_source: str = Form(...),
    trained_tokenizer_name: str = Form(...)
):
    tokenized_for_infer_file = tokenize_file(
                  filepath = file_to_tokenize_path, 
                  lang_source = lang_source,
                  trained_tokenizer_name=trained_tokenizer_name)    
    return reply_success(message = "Done", result=tokenized_for_infer_file)