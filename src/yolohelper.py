import os
from typing import List
# check Pillow version number
import PIL
print('Pillow Version:', PIL.__version__)
import numpy as np
import os
from pathlib import Path
import re
import json
import shutil
import matplotlib.pyplot as plt

# load and show an image with Pillow
from PIL import Image,ImageDraw
Image.MAX_IMAGE_PIXELS = None
import numpy as np
from enum import Enum


class ImageFileFormatEnum(Enum):
    jpeg = 1
    jpg = 2
    png = 3

def is_valid_yolo_data_directory(data_path:str, datatype:str):
    """Checks if a directory for YoloV4 is valid.

    Args:
        data_path (str): path to directory
        datatype (YoloDatatypeEnum): Type of data Train, Test or Validation

    Returns:
        path: valid path
    """
    datapath = os.path.join(data_path)

    assert os.path.exists(datapath), "The train directory %s does not exist. Please enter correct path." %data_path

    files = os.listdir(datapath)
    assert len(files) > 0, "The train directory should not be empty. This should contain images and corresponding annotations text files."

    return datapath

def prepare_yolo_files_list(imagefileformat:ImageFileFormatEnum, directory_path:str, yolo_data_type:str):
    """Extarct image file types by filtering required format

    Args:
        imagefileformat (ImageFileFormatEnum): Image file format to create the image file list for YoloV4 training.
        directory_path (str): path to Yolo Image data
        yolodatatype (YoloDatatypeEnum): Type of data Train, Test or Validation

    Returns:
        datafileslist: filtered files list with full path
    """
    datapath = is_valid_yolo_data_directory(directory_path, yolo_data_type)
    datafiles = [os.path.join(datapath, f) for f in os.listdir(datapath) if imagefileformat.name in f]
    assert len(datafiles) > 0, "There are no files matching the filetype %s in the directory %s." %(imagefileformat, datapath)
    
    return datafiles

def copy_yolo_files(sourcedir_datafiles:List, destination_path:str):

    targetpath = is_valid_yolo_data_directory(destination_path, "")
    assert len(sourcedir_datafiles) > 0, "The source files list should not be empty."

    donotexist = []
    for f in sourcedir_datafiles:
        if os.path.exists(f):
            shutil.copy(f, targetpath)
        else:
            donotexist.append(f)
    
    assert len(donotexist) == 0, "Some files could not be copied. Please check the source path."
    
    return True

def write_yolo_file(targetpath:str, textfilename:str, yoloimagefileslist:List):
    """Create Yolo data files with image file paths

    Args:
        targetpath (str): destination path
        textfilename (str): text file name. example: train.txt
        yoloimagefileslist (List): list of image files each with full path
    """
    assert os.path.exists(targetpath), "Target path to write file to does not exist."

    with open(os.path.join(targetpath,textfilename), "w") as f:
        f.write("\n".join(yoloimagefileslist))
    
    assert os.path.exists(os.path.join(targetpath,textfilename)), "The Yolo text file could not be created."

def plot_image(imagespath:List):
    for im in imagespath:
        img = Image.open(im)
        print(img.size)
        plt.imshow(img)
        plt.show()

def validate_locations(locationfiles:list):
    """Validate locations files for Yolo.

    Args:
        locationfiles (list): text files with full directory path
    """
    valid_count = 0
    for f in locationfiles:
        center_locs= []
        locs = {}
        with open(f, 'r') as reader:
            assert f, "File does not exist. Please check the text file path." %f

            print("****************************"+f+"*****************************")
            center_locs = reader.read().rstrip().split("\n")
            locs = [np.fromstring(loc, dtype=float, sep=' ').tolist() for loc in center_locs ]
            print("\n")
            print("locations for file "+f+" are ",locs)
            valid_count += 1
    
    assert len(locationfiles) == valid_count, "Some annotation text files could not be validated."
    return True

def plot_locs(impath:str, obj_loc_filepath:str, targetpath:str):
    """Plot bounding boxes from locations to validate input image, textfile data pair.

    Args:
        impath (str): path to image file
        obj_loc_filepath (str): path to object locations file
        targetpath (str): target path to save the image with drawing bounding boxes
    """
    img = Image.open(impath)
    assert img, "Image path is incorrect."

    draw = ImageDraw.Draw(img)
    filename = os.path.basename(impath)

    with open(obj_loc_filepath, 'r') as reader:
        content = reader.read().rstrip().split("\n")
        locs = [np.fromstring(object, dtype=float, sep=' ').tolist() for object in content ]
        if len(locs[0]) > 0:
            for loc in locs:
                if len(loc) > 0:
                    label, x,y,ow,oh = loc
                    w,h = img.size
                    x,y,ow,oh = int(x*w),int(y*h),int(ow*w),int(oh*h) #<x> = <absolute_x> / <image_width> or <height> = <absolute_height> / <image_height>
                    
                    #print("class",label,x,y,ow,oh )
                    draw.rectangle((x-ow,y-oh,x+ow,y+oh),outline=(255,0,0),width=2)
            img.convert("RGB").save(os.path.join(targetpath,filename))
            plt.imshow(img)
            plt.show()

def validate_yolo_input_data(datapath:str, targetpath:str):
    """This method with validate if there are image file, text file pairs for each image with corresponding annotations for Yolo.

    Args:
        datapath (str): path to data directory
    """

    datapath = is_valid_yolo_data_directory(datapath, "")
    images = [os.path.join(datapath, f) for f in os.listdir(datapath) if ".jpg" in f ]
    object_locations = [os.path.join(datapath, f) for f in os.listdir(datapath) if ".txt" in f]
    print(len(images), len(object_locations))
    assert len(images)>0 and len(object_locations) >0 and len(images) == len(object_locations), "Images/annotations files are not corect."
    images.sort()
    object_locations.sort()

    plot_image(images[:6])
    result= validate_locations(object_locations)

    assert result, "Validation of annotation text files failed."
    
    #plot locations for each location <object-class> <x_center> <y_center> <width> <height>

    plot_locs(images[0], object_locations[0], os.path.join(targetpath))

    return True


    


    