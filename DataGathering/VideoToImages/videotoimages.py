#
# This script cuts multiple videos into frames and puts labels them by their name
# Takes in an input folder, where the videos are
# And an output folder where the new pictures go
# Name your videos <label><number>.mp4
# Numbers will be cut from the name
#

import cv2
import os

#Cuts all numbers from string
def string_cut_numbers(string):
    return ''.join([i for i in string if not i.isdigit()])

#Loads all videos from folder to dictionary
#Dictionary's key is video files name(numbers and file type are cut away)
#Value is a list of the videos under the same name
def load_videos(input_folder):
    videos = os.listdir(input_folder)
    vids = {}
    for i in videos:
        s = i.split('.')
        try:
            if(s[1] == 'mp4'):
                name = string_cut_numbers(s[0])
                if(name in vids):
                    vids[name].append(cv2.VideoCapture(s[0] + '.' + s[1]))
                else:
                    vids[name] = [cv2.VideoCapture(s[0] + '.' + s[1])]
        except:
            print('skipping a file which is not mp4')
            continue
    return vids

# Cuts videos into frames
# Frames are sperated into folders, defined by video dictionary's key

def videos_to_pictures(output_folder,  vids):
    count = 1
    video_count = 0;
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    os.chdir(output_folder)
    for key,  value in vids.items():
        if not os.path.exists(key):
            os.makedirs(key)
        os.chdir(key)
        dir = os.listdir('.')
        count = len(dir)
        for vid in value:
            video_count += 1
            print(video_count + " videos processed")
            while True:
                success,  image = vid.read()
                if(cv2.waitKey(10) == 27 or success == False):
                    break
                cv2.imwrite(str(count) + '.bmp',  image)
                count += 1
        os.chdir('..')

def main():
    inputfolder = input('Give input folder: ')
    vids = load_videos(inputfolder)
    outputfolder = input('Give output folder: ')
    videos_to_pictures(outputfolder,  vids)

    
if __name__ == '__main__':
    main()
