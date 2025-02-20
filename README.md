# usage: translate between any 2 language using OpenNMT with little customized tokenizer
target: Vietnamese<br>
source: Khmer <br>
with result: <br>
no bpe tokenize (better than bpe) -> Khmer to Vietnamese -> train_acc : 70%, val_acc: 39% <br>
Experiment more (no bpe tokenize) -> Khmer to English -> train_acc : 90%, val_acc : 63% <br>
future solution : Khmer -> English -> Vietnamese (english to vietnam using MTeT repo)

converge at epoch 120k-140k

# prepare 
make a folder ./data
put file ./data/target_source.txt 
# usage: run this bash file in order
```
bash create_env.sh
```
```
bash run-all-no-bpe.sh
```
```
bash 8_translate-no-bpe.sh
```


# can skip this part (it explain what i extra did to get the code run)
(dont need to read this anymore) prepare env:
- env_1 (before train)
```
pip install khmer-nltk
pip install underthesea
pip install nltk
pip install numpy==1.25.0
```

- env_2(train)
dont need anymore cause I integrate into source code already: (build OpenMNT)
```
!wget https://github.com/OpenNMT/OpenNMT-py/archive/refs/tags/2.3.0.tar.gz
!tar -zxvf 2.3.0.tar.gz
!mv OpenNMT-py-2.3.0 OpenNMT-py
```
to use CLI command for OpenNMT-py
```
%cd OpenNMT-py
!pip install -e .
```
dont need this:
```
!pip install OpenNMT-py==2.3.0
```



- things I change w.r.t original OpenNMT:
    to enable training from pretrain<br>
    in OpenNMT-py/onmt/models/model_saver.py change (to bypass security safe)

    ```
    def load_checkpoint(ckpt_path):

        import torch
        from onmt.inputters.text_dataset import TextMultiField

        # Add TextMultiField to the allowed safe globals
        torch.serialization.add_safe_globals([TextMultiField])
        """Load checkpoint from `ckpt_path` if any else return `None`."""
        checkpoint = None
        if ckpt_path:
            logger.info('Loading checkpoint from %s' % ckpt_path)
            checkpoint = torch.load(ckpt_path,
                                    map_location=lambda storage, loc: storage, weights_only=False)
        return checkpoint
    ```


    to enable translate from checkpoint <br>
    in OpenNMT-py/onmt/model_builder.py add in line 81 (to bypass security safe)

    ```
    def load_test_model(opt, model_path=None):
        import torch
        from onmt.inputters.text_dataset import TextMultiField

        # Add TextMultiField to the allowed safe globals
        torch.serialization.add_safe_globals([TextMultiField])

        if model_path is None:
            model_path = opt.models[0]
        checkpoint = torch.load(model_path,
                                map_location=lambda storage, loc: storage, weights_only=False)
    ```

