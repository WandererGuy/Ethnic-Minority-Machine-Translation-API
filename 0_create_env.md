run this code to also prepare env , fix to your env path 
```
conda create -p /home/tnadmin/manhT04/text_to_text_translation/Ethnic-Minority-Machine-Translation-API/env python=3.10 -y
conda activate /home/tnadmin/manhT04/text_to_text_translation/Ethnic-Minority-Machine-Translation-API/env
tar -zxvf 2.3.0.tar.gz
mv OpenNMT-py-2.3.0 OpenNMT-py
cd OpenNMT-py
pip install -e .
cd ..
pip install fastapi uvicorn pydantic python-multipart
pip install transformers
pip install OpenNMT-tf
pip install tensorflow
pip install 'keras<3.0.0'
pip install mediapipe-model-maker --no-deps
pip install gdown
pip install pandas
```
