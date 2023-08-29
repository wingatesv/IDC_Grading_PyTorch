import requests
import zipfile
from pathlib import Path




def get_temp(data_path):
  data_path = Path(data_path)
  temp_path = data_path / 'templates'
  # download dir if not exist
  if temp_path.is_dir():
    print(f'{temp_path} already exists...skipping download')
  else:
    print(f'{temp_path} not exists, creating one....')
    temp_path.mkdir(parents=True, exist_ok=True)

    # download data
    with open(data_path / 'templates.zip', 'wb') as f:
      request = requests.get('https://github.com/wingatesv/SN_IDC_Grading/raw/main/Conventional%20SN/Template-20230228T123629Z-001.zip')
      print('downloading zip.....')
      f.write(request.content)

    # unzip zip file
    with zipfile.ZipFile(data_path/'templates.zip', 'r') as zip_ref:
      print('unzipping data...')
      zip_ref.extractall(temp_path)
