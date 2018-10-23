import sys, os
import tensorflow as tf
import subprocess

sys.path.append("models")
from models.BiSeNet import build_bisenet

SUPPORTED_MODELS = ["FC-DenseNet56","BiSeNet"]

SUPPORTED_FRONTENDS = ["ResNet50", "ResNet101", "ResNet152"]

def download_checkpoints(model_name):
    subprocess.check_output(["python", "utils/get_pretrained_checkpoints.py", "--model=" + model_name])


def build_model(model_name, net_input, num_classes, crop_width, crop_height, frontend="ResNet101", is_training=True):

	print("Preparing the model ...")

	if model_name not in SUPPORTED_MODELS:
		raise ValueError("The model you selected is not supported. The following models are currently supported: {0}".format(SUPPORTED_MODELS))

	if frontend not in SUPPORTED_FRONTENDS:
		raise ValueError("The frontend you selected is not supported. The following models are currently supported: {0}".format(SUPPORTED_FRONTENDS))

	if "ResNet50" == frontend and not os.path.isfile("models/resnet_v2_50.ckpt"):
	    download_checkpoints("ResNet50")
	if "ResNet101" == frontend and not os.path.isfile("models/resnet_v2_101.ckpt"):
	    download_checkpoints("ResNet101")
	network = None
	init_fn = None
	if model_name == "BiSeNet":
	# BiSeNet requires pre-trained ResNet weights
	   network, init_fn = build_bisenet(net_input, preset_model = model_name, frontend=frontend, num_classes=num_classes, is_training=is_training)
	else:
	    raise ValueError("Error: the model %d is not available. Try checking which models are available using the command python main.py --help")

	return network, init_fn
