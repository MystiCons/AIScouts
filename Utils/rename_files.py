import os

input_path = "./asd/"
output_path = "./Car/"

output_count = len(os.listdir(output_path))

count = 0
for file in os.listdir(input_path):
    os.rename(input_path + file, output_path + str(output_count + count) + '.bmp')
    count += 1
