# usage: translate between any 2 language using OpenNMT with sentencepiece tokenizer
target: Vietnamese<br>
source: Khmer <br>
with result: <br>
no bpe tokenize (better than bpe) -> Khmer to Vietnamese -> train_acc : 70%, val_acc: 39% <br>
Experiment more (no bpe tokenize) -> Khmer to English -> train_acc : 90%, val_acc : 63% <br>
future solution : Khmer -> English -> Vietnamese (english to vietnam using MTeT repo)

converge at epoch 120k-140k
# medium post
https://medium.com/@manhtech264/neural-machine-translation-between-any-2-languages-part-1-74980f50e3a6

# prepare env (better to run eacvh command in bash file manually)
```
bash 0_build_spm.sh
bash 0_create_env.sh
```
# run server 
```
python main.py
```
# usage 
use postman to send API to server , read about API in ./routers/infer.py
# dockerfile
U can run a demo use Docker by simple build image with Dockerfile then run the built image 
from the repo
```
docker build  --no-cache -t nmt_main .    
docker run -it -p 5021:5021 -v D:\MANH_T04:/app/Ethnic-Minority-Machine-Translation-API/checkpoints nmt_main
```

