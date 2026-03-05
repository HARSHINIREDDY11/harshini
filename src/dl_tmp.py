import zipfile
import os
import shutil

print('Extracting inner zip...')
with zipfile.ZipFile('trashnet-master/data/dataset-resized.zip', 'r') as z:
    z.extractall('.')

print('Moving...')
target='./trashnet'
if os.path.exists(target):
    shutil.rmtree(target, ignore_errors=True)
os.rename('dataset-resized', target)
print('Done!')
