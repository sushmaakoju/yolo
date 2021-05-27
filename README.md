# yolo

### This is repository for YoloV4 training for object detection for CGSI project at Robotics Institute, Carnegie Mellon University.

##### Please note that all the Python methods in the src folder are meant to be used for Jupyter Notebooks.
##### In fact the methods are picked up from notebooks to put in a single palce for re-usability or reference for later.  
##### Hence there is expected to be some hard coded values that work for this project or generally needs testing.
##### Due to this these methods, should be well tested to your target dataset, folder paths before using the methods.

### Preparation for Training
* Assumption: Training sub images and corresponding yolov4 annotation formatted text files were generated
* Decide on cross validation folds and generate train.txt and val.txt using 
    * https://github.com/sushmaakoju/yolo/blob/main/src/yolohelper.py 
* Generate data distribution across training and validation for each folds, 
    * fold-wise vehicle counts, single or multi class-wise distribution using methods in
    * https://github.com/sushmaakoju/yolo/blob/main/src/plots_for_dataset.py 
    * https://github.com/sushmaakoju/yolo/blob/main/src/yolohelper.py 
* The above plots on data help understand the noise and ground truth distribution between folds, whether single class or multi-class. This will help in analysis for evaluation.
* Use single obj/data folder to store all train image, annotation file pairs.
* For validation images: save all validation images to each of separate folders since –save_labels creates imagename.txt files during Test evaluation step (slide 20)
* Create obj_foldnumber.data for each of cross validation folds such that each includes corresponding train_foldnumber.txt and val_foldnumber.txt
* Update obj.names to class names that needs detection
* Generate anchors specific to each of train_fold you would like to train.:
    * ./darknet detector calc_anchors cross_validation_folds/anchors/cfg/obj2.data -num_of_clusters 9 -width 768 -height 768

### Steps to Train custom object detection : specific to Selwyn datasets:
* Clone https://github.com/AlexeyAB/darknet
* Download Yolov4-custom.cfg from https://github.com/AlexeyAB/darknet/blob/master/cfg/yolov4-custom.cfg 
* Download weights file i.e. yolov4.conv.137 from  https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.conv.137 
* Update width and height to each of training image size
* Update max_batches one for first 1000 iterations
* And another max_batches value for after 1000 iterations – should be rounded to closest to Total number of training images. (refere slide 10 for more elaborate description)
* Update classes = total number of classes for required for object detection. Use classes =1
* Update filters = (num of classes + 5) * 3, filters = 18 (for 1 class)
* Create obj.data, obj.names, backup folder (to save results).
* In obj.data, generate training images paths saved to train.txt and validation image paths saved to val.txt.
* Update obj.names path in obj.data and backup folder path.
* Generate anchors for obj.data configuration (consisting of train and validation text files with image paths)
    * ./darknet detector calc_anchors /home/sakoju/evaluate/cross_validation_folds/anchors/cfg/obj2.data -num_of_clusters 9 -width 768 -height 768
    * Update the anchors results saved (in darknet/ folder) to yolov4-custom.cfg
* For first 1000 iterations- use max_batches = 1000
* Update detector.c -> line 381 and as follows to save weights every 600 iterations. For “this” training specific case, we save weights every 600 iterations
    * ![image](https://user-images.githubusercontent.com/8979477/119887462-30882280-bf02-11eb-88ab-3af931d7cbc1.png)
* Execute training for above configurations:
    * ./darknet detector train cfg/obj.data cfg/yolov4-custom.cfg  yolov4.conv.137 -dont_show -mjpeg_port 8090 -map -gpus 0,1,2,3 | tee log.txt
* Copy weights generated for 600 iterations to a separate folder and re-train using this weights.
* Then for > 1000 iterations, by using saved weights file from backup folder from step 9, train with max_batches from step 5
    * ./darknet detector train cfg/obj.data cfg/yolov4-custom.cfg  new_weights/yolov4_600.weights -dont_show -mjpeg_port 8090 -map -gpus 0,1,2,3 | tee log.txt

### How-To: Intermediate Results and analysis during training
* Training results are stored in log.txt in /darknet folder
* Using log_parser.py, by giving input source folder containing log.txt and target folder to save plots, https://github.com/sushmaakoju/yolo/blob/main/src/yolo_plots.py 
* Generate Avg loss plots over iterations

### Steps to Evaluate Map and Test
* For each of obj_foldnumber.data, map evaluation needs to be conducted or depending which obj.data or train, validation set you used.
* Update or use the bash script to run evaluations: 
    * https://github.com/sushmaakoju/yolo/blob/main/scripts/evaluate_map.sh 
* Map Evaluation commands used:
    * -dont_show (don’t show loss/results window)
    * -ext_output (save coordinates detected )
    * < cross_validation_folds/training/val3.txt > (input validation images)
    * map_results3.txt (output log file to store command line output)
* Update or use the back script to run test evaluations:
    * https://github.com/sushmaakoju/yolo/blob/main/scripts/evaluate_test_yolov4.sh 
* Test evaluation commands use: 
    * -thresh 0.25 (threshold for ioU)
    * -dont_show (don’t show loss/results window)
    * -save_labels  (saves the labels as imagename.txt)
    * < cross_validation_folds/training/val7.txt  (input validation images)
    * > test_results7.txt (output log file to store command line output)

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