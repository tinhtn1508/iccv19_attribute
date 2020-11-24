import os
import subprocess

annotationName = "/home/tinhtn/Downloads/PA-100K-20201008T181600Z-001/PA-100K/annotation.zip"
dataName = "/home/tinhtn/Downloads/PA-100K-20201008T181600Z-001/PA-100K/data.zip"

os.system("unzip {} -d {}".format(annotationName, "./dataset"))
os.system("unzip {} -d {}".format(dataName, "./dataset"))