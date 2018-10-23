import subprocess
import argparse
import os
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="ALL", help='Which model weights to download')
args = parser.parse_args()

if args.model == "ResNet50" or args.model == "ALL":
	subprocess.check_output(['wget','https://s3.amazonaws.com/pretrained-weights/resnet_v2_50.ckpt', "-P", "models"])

if args.model == "ResNet101" or args.model == "ALL":
	subprocess.check_output(['wget','https://s3.amazonaws.com/pretrained-weights/resnet_v2_101.ckpt', "-P", "models"])

if args.model == "ResNet152" or args.model == "ALL":
	subprocess.check_output(['wget','https://s3.amazonaws.com/pretrained-weights/resnet_v2_152.ckpt', "-P", "models"])

