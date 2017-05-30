import cv2
from fractions import  Fraction

crop_size_width = 128
crop_size_height = 128
image_aspect = [16, 9]
image_width = 0
image_height = 0
image = None
move_ratio = 16


def search_image():
    curr_position = [0, 0]
    while(True):
        while(True):
            crop = image[curr_position[1]:curr_position[1]+crop_size_height,  curr_position[0]: curr_position[0]+crop_size_width]
            if not(predict(crop)):
                curr_position[0] = int(curr_position[0] + move_ratio)
            else:
                curr_position[0] = curr_position[0] + crop_size_width
            if curr_position[0] + crop_size_width >= image_width:
                break
        curr_position[1] = curr_position[1] + crop_size_height
        curr_position[0] = 0
        if curr_position[1] + crop_size_height >= image_height:
            break


def predict(crop):
    cv2.imshow('cropped',  crop )
    cv2.waitKey()
    return False
    

def init():
    global image_width
    global image_height
    global crop_size_width
    global crop_size_height
    global image
    img_file = input('Image file location: ')
    image = cv2.imread(img_file)
    image_height, image_width, channels = image.shape
    fraction = Fraction(image_width,  image_height)
    image_aspect[0] = fraction.numerator
    image_aspect[1] = fraction.denominator
    crop_size_width = int(input('Crop width size (Must be divisible by given images aspect width [' + str(image_aspect[0]) + ']): ' ))
    while not(crop_size_width % image_aspect[0] == 0):
        print("Not divisible by aspect ratio")
        crop_size_width = int(input('Crop width size (Must be divisible by given images aspect width [' + str(image_aspect[0]) + ']): '))
    crop_size_height = int((image_aspect[1] * crop_size_width) / image_aspect[0])

    
def main():
    init()
    search_image()
    
    
if __name__ == '__main__':
    main()
