

import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import pickle

def compare_class_counts(train_filespath:str, anchor_files_path:str):
    ifiles = [os.path.join(train_filespath, f) for f in os.listdir(train_filespath) if ".txt" in f and "train" in f ]
    ifiles.sort()

    anchors_path = os.path.join(anchor_files_path)
    anchorfiles = [os.path.join(anchors_path, f) for f in os.listdir(anchors_path) if ".txt" in f and "anchors" in f ]
    countsper_class_files = [os.path.join(anchors_path, f) for f in os.listdir(anchors_path) if ".txt" in f and "counters" in f ]
    countsper_class_files.sort()
    len(anchorfiles), len(countsper_class_files)

    anchor_file_class_count = {}
    for file in countsper_class_files:
        with open(file, 'r') as filereader:
            counters_per_class_text = filereader.read().rstrip().split("\n")[0]
            anchor_file_class_count[os.path.basename(file)] = int(counters_per_class_text.split(" ")[2])
    vehicle_count = 0
    train_file_vehicle_couns = {}
    # for each train files list
    for trainfile in ifiles:
        vehicle_count = 0
        #open the train fold containing the train images paths 
        with open(trainfile, 'r') as filereader:
            files = filereader.read().rstrip().split("\n")
            print(len(files))
            for i in files:

                a = i.replace(".jpg",".txt")
                #open the annotation file corresponding to the train image file
                with open(a, 'r') as reader:
                    center_locs = reader.read().rstrip().split("\n")
                    #print(center_locs)
                    locs = [np.fromstring(loc, dtype=float, sep=' ').tolist() for loc in center_locs ]
                    #print(len(locs))
                    if len(locs[0]) > 0:
                        vehicle_count += len(locs)
        train_file_vehicle_couns[os.path.basename(trainfile)] = vehicle_count

    values = list(anchor_file_class_count.values())[2:]
    trainfolds = list(train_file_vehicle_couns.keys())[2:]
    plt.figure(figsize = (15,8))

    plt.bar(trainfolds, values)
    j = 2000
    # Add annotation to bars
    for i in range(len(trainfolds)):
        plt.annotate(values[i], (-0.1 + i, values[i] + j))
    plt.xlabel("Cross validation fold")
    plt.ylabel("Collapsed vehicle counts")
    plt.title("Distribution of vehicle counts in each fold training folds with validation for img group 3 until img group 9")
    plt.show()

def plot_classwise_distribution(annotation_pkl_file_path:str):
    assert annotation_pkl_file_path, "Path is invalid."

    afiles = [os.path.join(annotation_pkl_file_path, f) for f in os.listdir(annotation_pkl_file_path)]

    result = []

    #for each annotation pkl file
    for f in afiles:
        filename = os.path.splitext(os.path.basename(f))[0]
        #print(filename)
        with open(f, 'rb') as file:
            res = pickle.load(file)
            #get all class, get absolute locations, width and height
            for k,values in res.items():
                #get length of values, just need the count of annotations per class
                vals = values.tolist()
                if len(vals) > 0:
                    for v in vals:
                        result.append({'fileid':filename, "vehicle_class":k,"values":v})
                else:
                    result.append({'fileid':filename, "vehicle_class":k,"values":np.nan})
    
    df = pd.DataFrame(result)
    classes = df["vehicle_class"].unique().tolist()
    classes_counts = {}
    temp = df.copy()
    temp['values'] = df['values'].fillna(0)
    for clas in classes:
        this_class = temp[temp['vehicle_class'] == clas]
        count = this_class[this_class['values']!=0].count()['values']
        classes_counts[clas] = count
    
    plt.figure(figsize=(15,5))
    plt.bar(range(1,len(classes_counts)+1,1), list(classes_counts.values()), align='center')
    plt.xticks(range(1,len(classes_counts)+1,1), list(classes_counts.keys()))
    plt.title("Class-wise vehicle counts for sampled images with annotations")
    for x,y in zip(range(1,len(classes_counts)+1,1),list(classes_counts.values())):

        label = "{:d}".format(y)

        plt.annotate(label, # this is the text
                    (x,y), # this is the data label to plot
                    textcoords="offset points", # offset the points
                    xytext=(0,10), # distance from text to points (x,y)
                    ha='center') # horizontal alignment can be left, right or center
    int(max(classes_counts.values())*110/100.)
    plt.ylim(0,int(max(classes_counts.values())*110/100.))
    plt.xlim(0,len(classes_counts)+1)
    plt.show()