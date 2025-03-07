


def split_data(path):
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
        if (n < 5000):
            src_val.write(data[-1])
            tgt_val.write(data[0] + '\n')
        elif (n < 110000):
            src_train.write(data[-1])
            tgt_train.write(data[0]  + '\n')
        else:
            src_test.write(data[-1])
            tgt_test.write(data[0] + '\n')
        l += 1
        n += 1

    print('Số lượng dòng tổng cộng là ', l)
    print('Số lượng dòng đúng quy tắc có thể tách là ', n)
    
import os 
os.makedirs('./data', exist_ok=True)
path = './data/target_source.txt'
split_data(path)