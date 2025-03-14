import requests

url = "http://127.0.0.1:4013/translate_en"

payload = {'language': 'en',
'inputs': 'Our teams aspire to make discoveries that impact everyone, and '}
files=[

]
headers = {}

response = requests.request("POST", url, headers=headers, data=payload, files=files)

print(response.text)
