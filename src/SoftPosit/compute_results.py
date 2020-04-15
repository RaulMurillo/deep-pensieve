#!/usr/bin/python

import sys
import csv
import os
from os import listdir
from os.path import isfile, join

mypath = sys.argv[1]
total_imgs = 0
total_top5 = 0
total_acc  = 0

onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
for f in onlyfiles:
    name = f.split('.')[0].split('_')
    images = int(name[2]) - int(name[1])
    acc = 0
    top5 = 0
    with open(mypath + f, 'r') as csvfile:
        reader = csv.reader(csvfile)
        line = 0
        for row in reader:
            if line > 0:
                acc  = float(row[0])
                top5 = float(row[1])
            line += 1
    total_imgs += images
    total_top5 += int(top5*images)
    total_acc  += int(acc*images)

print(f"Total images: {total_imgs}")

print(f"Validation Accuracy: {total_acc/total_imgs}")
print(f"Validation Top-5: {total_top5/total_imgs}")

# Save training results
hist = {}
hist["val_acc"] = (total_acc/total_imgs)
hist["top5"] = (total_top5/total_imgs)

zd = zip(hist.values())

with open(mypath + 'posit8_quire.csv', 'w') as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerow(hist.keys())
    writer.writerows(zd)
