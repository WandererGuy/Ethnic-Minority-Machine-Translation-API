import subprocess            
import time 

"""
PUT target_source.txt file in ./data
Activate environment
run this script 
"""
import os
import shutil
import nltk
nltk.download('punkt_tab')

if os.path.exists("OpenNMT-py/onmt/model_builder.py"):
    os.remove("OpenNMT-py/onmt/model_builder.py")
shutil.copy("OpenNMT_replace/model_builder.py", "OpenNMT-py/onmt")

def running_python(command):
    print ("************************************")
    print (" ".join(command))
    # Running the subprocess with the provided command
    subprocess.run(command, text=True)
    time.sleep(4)

def running_bash_file(command):
    print ("************************************")
    print (" ".join(command))
    # Running the subprocess with the provided command
    subprocess.run(command, text=True)
    time.sleep(4)

def train_source_spm(vocab_size):
    command = f"spm_train --input=data/source.txt --model_prefix=source --vocab_size={vocab_size} --character_coverage=1.0 --model_type=unigram"
    try:
        subprocess.run(command, shell=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        return False
def train_target_spm(vocab_size):
    command = f"spm_train --input=data/target.txt --model_prefix=target --vocab_size={vocab_size} --character_coverage=1.0 --model_type=unigram"
    try:
        subprocess.run(command, shell=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        return False


def check_env():
    tool_ls = ["spm_train", "onmt_build_vocab", "onmt_train"]
    for tool in tool_ls:
        command = f"{tool} --help"
        print ("running command: ", command)
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        
        if result.stdout:
            pass
        else:
            raise Exception(f"please install {tool}")
if __name__ == "__main__":

    check_env()

    # train tokenizer
    command = ["python", "1_1_train_spm_split.py"]
    running_python(command)

    for vocab_size in [8000, 4000, 2000]:
        if train_source_spm(vocab_size) == False:
            print ("******************** reduce vocab size ********************")
            time.sleep(5)
            continue 
        else:
            break

    for vocab_size in [8000, 4000, 2000]:
        if train_target_spm(vocab_size) == False:
            print ("******************** reduce vocab size ********************")
            time.sleep(5)
            continue 
        else:
            break

    command = ["python", "1_4_check_prepare.py"]
    running_python(command)

    command = ["python", "2_split.py"]
    running_python(command)

    command = ["python", "3_tokenize_source.py"]
    running_python(command)

    command = ["python", "4_tokenize_target.py"]
    running_python(command)

    command = ["python", "5_config.py"]
    running_python(command)

    command = ["bash", "6_train-no-bpe.sh"]
    running_bash_file(command)