from PIL import Image
import os
import tqdm

path = './'
output = './'
for file in tqdm(os.listdir(path)):
    try:
        im = Image.open(path + file)
        dirlen = len(os.listdir(output))
        im.save(output + str(dirlen) + '.jpg')
    except Exception as e:
        print('Failed to convert image')

