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

tgt_train = open('./data/tgt-train.txt', mode='r', encoding='utf8')
tgt_val = open('./data/tgt-val.txt', mode='r', encoding='utf8')
tgt_test = open('./data/tgt-test.txt', mode='r', encoding='utf8')

# tgt_train_token = open('./data/tgt-train-token.txt', mode='w+', encoding='utf8')
# tgt_val_token = open('./data/tgt-val-token.txt', mode='w+', encoding='utf8')
# tgt_test_token = open('./data/tgt-test-token.txt', mode='w+', encoding='utf8')

tokenize_file('./data/tgt-train.txt', './data/tgt-train-token.txt')
tokenize_file('./data/tgt-val.txt', './data/tgt-val-token.txt')
tokenize_file('./data/tgt-test.txt', './data/tgt-test-token.txt')

n = 0
for line in tgt_train:
    n+=1
print('Số lượng câu trong tập huấn luyện đích là ', n)
n = 0
for line in tgt_val:
    n+=1
print('Số lượng câu trong tập thẩm định đích là ', n)
n = 0
for line in tgt_test:
    n+=1
print('Số lượng câu trong tập kiểm tra đích là ', n)
print("Tokenize sucess")
