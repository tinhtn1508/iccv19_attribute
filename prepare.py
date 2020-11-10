import os
import subprocess

annotationName = "/content/mydrive/My\ Drive/dataset/pa100k/annotation.zip"
dataName = "/content/mydrive/My\ Drive/dataset/pa100k/data.zip"

os.system("unzip {} -d {}".format(annotationName, "./dataset"))
os.system("unzip {} -d {}".format(dataName, "./dataset"))