# Utilities

## data_manipulation.py
Dependancies:    

 * OpenCV ([Installation script](https://github.com/MystiCons/AIScouts/blob/master/InstallScripts/OpenCV_install_script.sh))    
 * numpy    
 * tqdm    

data_manipulation.py was used in the project to create artificial data and seperate images automatically from each other to create labeled data.    

The methods will output only grayscale images. 

The scripts has 3 functionalities:   
1. Cluster images and copy them to different folders.   
2. Color Quantization (Reduces shades of the colors)    
3. Flip images horizontally    

Quick example:    
```
# Where the images are
path = "/home/user/images/"

# Create an instance of the manipulator with output folder for clustering
manipulator = DataManipulation(path + 'output/')

# Cluster images to 3 categories
manipulator.try_cluster_training_data(path, 3)

# Reduces the shades of the images in the given folder to 8 resizes to 128*128 and saves into save_images_path
manipulator.color_quantization(path + 'output/', 8, 128, save_images_path=path + 'output/')

# Flips the images horizontally and resizes to 128*128 and saves to flipped_images folder in path/output
manipulator.flip_images(path + 'output/', path + 'output/' + "flipped_images/", 128)

```

## Vpn

The vpn folder has some utilities which were used to create a pptp vpn connection to labranet and keep it alive. Also startup script which will launch parking detection when the system boots.

the "labranet" file requires username and password to be inputted and you should change the paths in the files.

## rename_files.py
This script simply moves input folders files to output folder and renames them according to the output folders file count. 
