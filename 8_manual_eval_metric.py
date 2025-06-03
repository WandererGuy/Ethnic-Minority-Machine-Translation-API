
# """#!/bin/bash
# perl OpenNMT-py/tools/multi-bleu-detok.perl ./data/fake-tgt-test.txt < ./data/fake-pred-no-bpe-detokenize.txt
# """

import subprocess
def send_request(file_upload_path):
    import requests

    url = "http://0.0.0.0:5021/translate-language-for-eval"

    payload = {'file_upload_path': file_upload_path,
    'model_checkpoint_path': '/home/manh264/code_linux/NMT_server/src/checkpoints/vie_2M_checkpoint.pt',
    'source_checkpoint_tokenizer_path': '/home/manh264/code_linux/NMT_server/src/checkpoints/vie_2M_source.model',
    'target_checkpoint_tokenizer_path': '/home/manh264/code_linux/NMT_server/src/checkpoints/vie_2M_target.model'}
    files=[

    ]
    headers = {}

    response = requests.request("POST", url, headers=headers, data=payload, files=files)

    print(response.text)
    json_data = response.json()
    return json_data["result"]

def fix_windows_path(path):
    path = path.replace("\\", "/")
    if "C:" in path:
        path = path.replace("C:", "/mnt/c")
    elif "D:" in path:
        path = path.replace("D:", "/mnt/d")
    return path

"""
U need :
1 source file in vietnamese
1 groundtruth file translated in english 
this script takes 
1 source file in vietnamese -> predict file in english
generate command to calculate BLEU between groundtruth file and predict file
"""
source_file = r"/home/manh264/code_linux/NMT_server/src/prepare_eval_dataset_2/source_vi.txt"
predict_file = send_request(file_upload_path = source_file)
groundtruth_file = r"/home/manh264/code_linux/NMT_server/src/prepare_eval_dataset_2/groundtruth_en.txt"


groundtruth_file = fix_windows_path(groundtruth_file)
predict_file = fix_windows_path(predict_file)


# Command and arguments
cmd = [
    "perl",
    "OpenNMT-py/tools/multi-bleu-detok.perl",
    groundtruth_file, 
    "<",
    predict_file
]
print ("predict_file", predict_file)
print ("groundtruth_file", groundtruth_file)
print ("please run command manually")
print (" ".join(cmd))

