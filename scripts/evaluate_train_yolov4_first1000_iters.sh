#!/bin/bash
echo "This script is for Yolo training for custom object detection."
echo "This script assumes you modified the detector.c file in darknet/src folder to save weights every 600 iterations."
echo "This script assumes you have updated max_batches=1000 and learning rate = 0.0001."
echo "This script assumes you are using single GPU to run first 1000 iterations."
echo "We use seperate folders for each of cross validation folds' weights so the saved weights files are not overwritten accidentally."

read -n 1 -s -r -p "If above assumptions are met, press any key to continue or press CTRL+C to cancel this script."

echo "The present working directory is `pwd`"
DIR="$(dirname "${BASH_SOURCE[0]}")"  # Get the directory name
DIR="$(realpath "${DIR}")" 
echo "The present working directory is ${DIR}"

if [ -d ${DIR} ] 
then
    echo "Directory ${DIR} exists." 
else
    echo "Error: Directory ${DIR} does not exists."
    exit 9999 #exit with 9999 error code
fi

echo "This script assumes you have provided path to obj folder with images in your obj.data file."
echo "Starting training for first 1000 iterations for train fold 3"
./darknet detector train cross_validation_folds/training/cfg/obj3.data cross_validation_folds/training/cfg/yolov4-custom_cv3.cfg yolov4.conv.137 -dont_show -mjpeg_port 8090 -map | tee log_cv_fold3.txt
echo "Starting training for first 1000 iterations for train fold 4"
./darknet detector train cross_validation_folds/training/cfg/obj4.data cross_validation_folds/training/cfg/yolov4-custom_cv4.cfg yolov4.conv.137 -dont_show -mjpeg_port 8090 -map | tee log_cv_fold4.txt

echo "Starting training for first 1000 iterations for train fold 5"
./darknet detector train cross_validation_folds/training/cfg/obj5.data cross_validation_folds/training/cfg/yolov4-custom_cv5.cfg yolov4.conv.137 -dont_show -mjpeg_port 8090 -map | tee log_cv_fold5.txt

echo "Starting training for first 1000 iterations for train fold 6"
./darknet detector train cross_validation_folds/training/cfg/obj6.data cross_validation_folds/training/cfg/yolov4-custom_cv6.cfg yolov4.conv.137 -dont_show -mjpeg_port 8090 -map | tee log_cv_fold6.txt

echo "Starting training for first 1000 iterations for train fold 7"
./darknet detector train cross_validation_folds/training/cfg/obj7.data cross_validation_folds/training/cfg/yolov4-custom_cv7.cfg yolov4.conv.137 -dont_show -mjpeg_port 8090 -map | tee log_cv_fold7.txt

echo "Starting training for first 1000 iterations for train fold 8"
./darknet detector train cross_validation_folds/training/cfg/obj8.data cross_validation_folds/training/cfg/yolov4-custom_cv8.cfg yolov4.conv.137 -dont_show -mjpeg_port 8090 -map | tee log_cv_fold8.txt

echo "Starting training for first 1000 iterations for train fold 9"
./darknet detector train cross_validation_folds/training/cfg/obj9.data cross_validation_folds/training/cfg/yolov4-custom_cv9.cfg yolov4.conv.137 -dont_show -mjpeg_port 8090 -map | tee log_cv_fold9.txt


echo "Training first 1000 iterations is completed."