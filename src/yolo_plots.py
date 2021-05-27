import cv2
import numpy as np
import sys
import argparse
import os
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
from pylab import *
import numpy as np
import logging
import platform
import re
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

def plot_precision_recall(input_file:str):
    output_file = input_file.replace(".txt",".png")
    f = open(input_file) 
    lines = [line.rstrip("\n") for line in f.readlines()]

    iters = []
    precisions = []
    recalls = []
    
    for line in lines:
        cols = line.split()
        if len(cols)<3:
            continue
        iters.append(float(cols[0][:-1]))
        precisions.append(float(cols[1]))
        recalls.append(float(cols[2]))

    print("Iterations",iters)
    print("Precision valies", precisions)
    print("Recall values" ,recalls)

    fig= plt.figure()
    ax = fig.add_subplot(111)

    plt.plot(iters,precisions,label = "precision")
    plt.plot(iters,recalls, label = "recall")
    plt.legend()
    ax.set_yticks(np.linspace(0,1,11))
    plt.grid()
    plt.xlabel("number of iterations in K(1.0 means 1000)")
    plt.ylabel("precision/recall value. Ideally should be 1")    
    #plt.figtext(0.95,0.9, "High recall means detect most of the fires cases. \n Low precision means a lot of misalarm")
    savefig(output_file)
    plt.show()

def visualize_anchors(anchor_dir:str):
    if not anchor_dir:
        anchor_dir = os.path.join(r'C:\Users\exx\Documents\cg-for-synthetic-images\task5-yolov4\yolo\anchors')


    print ("anchors list you provided{}".format(anchor_dir))

    [H,W] = (768,768)
    stride = 32

    cv2.namedWindow('Image')
    cv2.moveWindow('Image',100,100)

    colors = [(255,0,0),(255,255,0),(0,255,0),(0,0,255),(0,255,255),(55,0,0),(255,55,0),(0,55,0),(0,0,25),(0,255,55)]

    anchor_files = [f for f in listdir(anchor_dir) if (join(anchor_dir, f)).endswith('.txt')]
    for anchor_file in anchor_files:
        blank_image = np.zeros((H,W,3),np.uint8)
        

        f = open(join(anchor_dir,anchor_file))
        line = f.readline().rstrip('\n')

        anchors = line.split(',  ')

        filename = join(anchor_dir,anchor_file).replace('.txt','.png')
        
        print (filename)

        stride_h = 10
        stride_w = 3

        for i in range(len(anchors)):
            print(anchors[i].split(', '))
            print(map(float,anchors[i].split(', ')))
            (w,h) = map(float,anchors[i].split(', '))


            w=int (w*stride)
            h=int(h*stride)
            print (w,h)
            offset_x = 10+i*stride_w # this offset is just to make sure starting coordinates of anchors do not overlap each other
            offset_y = 10+i*stride_h
            
            cv2.rectangle(blank_image,(offset_x,offset_y),(offset_x+w,offset_y+h),colors[i])

            #cv2.imshow('Image',blank_image)

                
            cv2.imwrite(filename,blank_image)
            #cv2.waitKey(10000)

def get_file_name_and_ext(filename):
    (file_path, temp_filename) = os.path.split(filename)
    (file_name, file_ext) = os.path.splitext(temp_filename)
    return file_name, file_ext


def show_message(message, stop=False):
    print(message)
    if stop:
        sys.exit(0)

def log_parser(source_dir, save_dir, log_file, show_plot):

    assert os.path.exists(source_dir) ,"Source directory path does not exist."
    assert os.path.exists(save_dir),"Output directory path does not exist."
    assert log_file, "Log filename is empty."

    log_path = os.path.join(source_dir, log_file)

    print(log_path)
    file_name, _ = get_file_name_and_ext(log_path)
    log_content = open(log_path).read()

    iterations = []
    losses = []
    fig, ax = plt.subplots()
    # set area we focus on
    ax.set_ylim(0, 8)

    major_locator = MultipleLocator()
    minor_locator = MultipleLocator(0.5)
    ax.yaxis.set_major_locator(major_locator)
    ax.yaxis.set_minor_locator(minor_locator)
    ax.yaxis.grid(True, which='minor')

    pattern = re.compile(r"([\d].*): .*?, (.*?) avg")
    matches = pattern.findall(log_content)
    counter = 0
    log_count = len(matches)

    csv_path = os.path.join(save_dir, file_name + '.csv')
    out_file = open(csv_path, 'w')

    for match in matches:
        counter += 1
        if log_count > 200:
            if counter % 200 == 0:
                print('parsing {}/{}'.format(counter, log_count))
        else:
            print('parsing {}/{}'.format(counter, log_count))
        iteration, loss = match
        iterations.append(int(iteration))
        losses.append(float(loss))
        out_file.write(iteration + ',' + loss + '\n')

    ax.plot(iterations, losses)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.tight_layout()

    # saved as svg
    save_path = os.path.join(save_dir, file_name + '.svg')
    plt.savefig(save_path, dpi=300, format="svg")
    if show_plot:
        plt.show()

