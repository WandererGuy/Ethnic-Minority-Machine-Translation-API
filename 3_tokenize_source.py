import subprocess


def tokenize_file(input_file, output_file):
    model_name = "source"
    command = ["spm_encode",
                f"--model={model_name}.model",
                "--input", input_file, 
                "--output", output_file]
                
    print (" ".join(command))
    # Running the subprocess with the provided command
    result = subprocess.run(command, capture_output=True, text=True)

src_train = open('./data/src-train.txt', mode='r', encoding='utf8')
src_val = open('./data/src-val.txt', mode='r', encoding='utf8')
src_test = open('./data/src-test.txt', mode='r', encoding='utf8')

# src_train_token = open('./data/src-train-token.txt', mode='w+', encoding='utf8')
# src_val_token = open('./data/src-val-token.txt', mode='w+', encoding='utf8')
# src_test_token = open('./data/src-test-token.txt', mode='w+', encoding='utf8')

tokenize_file('./data/src-train.txt', './data/src-train-token.txt')
tokenize_file('./data/src-val.txt', './data/src-val-token.txt')
tokenize_file('./data/src-test.txt', './data/src-test-token.txt')

n = 0
for line in src_train:
    n+=1
print('Số lượng câu trong tập huấn luyện đích là ', n)
n = 0
for line in src_val:
    n+=1
print('Số lượng câu trong tập thẩm định đích là ', n)
n = 0
for line in src_test:
    n+=1
print('Số lượng câu trong tập kiểm tra đích là ', n)
print("Tokenize sucess")
