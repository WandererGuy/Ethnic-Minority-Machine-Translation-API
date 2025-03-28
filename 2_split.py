


def split_data(path, train_num, val_num, test_num):
    vn_kmer_txt = open(path, mode='r', encoding='utf8')
    src_train = open('./data/src-train.txt', mode='w+', encoding='utf8')
    tgt_train = open('./data/tgt-train.txt', mode='w+', encoding='utf8')
    src_val = open('./data/src-val.txt', mode='w+', encoding='utf8')
    tgt_val = open('./data/tgt-val.txt', mode='w+', encoding='utf8')
    src_test = open('./data/src-test.txt', mode='w+', encoding='utf8')
    tgt_test = open('./data/tgt-test.txt', mode='w+', encoding='utf8')

    if (vn_kmer_txt != None):
        print('Mở thành công file dữ liệu')

    l = 0
    n = 0

    for line in vn_kmer_txt:
        data = line.split('\t')
        if (len(data) < 2):
            l += 1
            continue
        if (n < val_num):
            src_val.write(data[-1])
            tgt_val.write(data[0] + '\n')
        elif (n < train_num + val_num):
            src_train.write(data[-1])
            tgt_train.write(data[0]  + '\n')
        else:
            src_test.write(data[-1])
            tgt_test.write(data[0] + '\n')
        l += 1
        n += 1

    print('Số lượng dòng tổng cộng là ', l)
    print('Số lượng dòng đúng quy tắc có thể tách là ', n)
    print ('Số lượng dòng trong train ', train_num)
    print ('Số lượng dòng trong val', val_num)
    print ('Số lượng dòng trong test ', test_num)
    
import os 
os.makedirs('./data', exist_ok=True)
path = './data/target_source.txt'
import random
random.seed(42)

# Read the lines of the file
with open(path, 'r', encoding='utf-8') as file:
    lines = file.readlines()

# Shuffle the lines
random.shuffle(lines)

# Write the shuffled lines back to a file (or overwrite the same file)
with open(path, 'w', encoding='utf-8') as file:
    file.writelines(lines)

with open(path, mode='r', encoding='utf8') as f:
    lines = f.readlines()
    total_num = len(lines)
    train_num = int(total_num * 0.8)
    val_num = int(total_num * 0.1)
    test_num = total_num - train_num - val_num

split_data(path, train_num, val_num, test_num)