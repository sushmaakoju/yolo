
#!/bin/bash
echo "This script is for Yolo Evaluating the Mean average precision"

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
echo "Starting mAP evaluation for train fold 3"

./darknet detector map cross_validation_folds/training/cfg/obj3.data cross_validation_folds/training/cfg/yolov4.cfg pre-trained-weights/yolov4.weights -dont_show -ext_output < cross_validation_folds/training/val3.txt > map_results3.txt

echo "Starting mAP evaluation for train fold 4"
./darknet detector map cross_validation_folds/training/cfg/obj4.data cross_validation_folds/training/cfg/yolov4.cfg pre-trained-weights/yolov4.weights -dont_show -ext_output < cross_validation_folds/training/val4.txt > map_results4.txt

echo "Starting mAP evaluation for train fold 5"
./darknet detector map cross_validation_folds/training/cfg/obj5.data cross_validation_folds/training/cfg/yolov4.cfg pre-trained-weights/yolov4.weights -dont_show -ext_output -gpus 0,1,2,3 < cross_validation_folds/training/val5.txt > map_results5.txt

echo "Starting mAP evaluation for train fold 6"
./darknet detector map cross_validation_folds/training/cfg/obj6.data cross_validation_folds/training/cfg/yolov4.cfg pre-trained-weights/yolov4.weights -dont_show -ext_output < cross_validation_folds/training/val6.txt > map_results6.txt

echo "Starting mAP evaluation for train fold 7"
./darknet detector map cross_validation_folds/training/cfg/obj7.data cross_validation_folds/training/cfg/yolov4.cfg pre-trained-weights/yolov4.weights -dont_show -ext_output < cross_validation_folds/training/val7.txt > map_results7.txt

echo "Starting mAP evaluation for train fold 8"
./darknet detector map cross_validation_folds/training/cfg/obj8.data cross_validation_folds/training/cfg/yolov4.cfg pre-trained-weights/yolov4.weights -dont_show -ext_output < cross_validation_folds/training/val8.txt > map_results8.txt

echo "Starting mAP evaluation for train fold 9"
./darknet detector map cross_validation_folds/training/cfg/obj9.data cross_validation_folds/training/cfg/yolov4.cfg pre-trained-weights/yolov4.weights -dont_show -ext_output < cross_validation_folds/training/val9.txt > map_results9.txt


echo "mAP evaluation is completed."
