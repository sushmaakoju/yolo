# yolo

### This is repository for YoloV4 training for object detection

### todo: add documentation

### About configuration: in yolov4_custom.cfg 
#### Max_batches = Number of classes * 2000
* number of classes = 1
* max_batches could be 2000 * number of classes, but should satisfy follwoing two conditions:
    * max_batches greater than or equal to total # of training images and 
    * max_batches greater than or equal to 6000
* For the "example" train set: image group 2 and validation : image group 1: 
* For this example small batch of training group 2, Total # of images in training set : 64426. 
* batch size = 64. 
* One epoch  requires number of train images / batch size => 64426/64 = ~1007 iterations. 
* one single epoch does full pass on a complete dataset with ~1007 iterations. so max_batches / iterations = total # of epochs. 
* so max batches should atleast be 64426 which meets all 2 conditions. 
* Update steps count which will be 80% and 90% of max_batches
* The anchor boxes are the “Prior” boxes to the training giving an intuition to look for objects that fit those sizes.

##### Command to generate anchors for given train data which is provided in obj.data file.
* ./darknet detector calc_anchors cross_validation_folds/anchors/cfg/obj2.data -num_of_clusters 9 -width 768 -height 768

##### Command to calculate map anchors for given train data which is provided in obj.data file, given yolo_config file and weights.
* ./darknet detector map cross_validation_folds/training/cfg/obj9.data cross_validation_folds/training/cfg/yolov4.cfg pre-trained-weights/yolov4.weights -dont_show -ext_output < cross_validation_folds/training/val9.txt > map_results9.txt

##### Command to generate anchors for given train data which is provided in obj.data file, given yolo_config file and weights.
* ./darknet detector test  cross_validation_folds/training/cfg/obj8.data cross_validation_folds/training/cfg/yolov4.cfg pre-trained-weights/yolov4.weights -dont_show -save_labels < cross_validation_folds/training/val8.txt > map_results8.txt