import os,cv2
import tensorflow as tf
import numpy as np
import time
from utils import utils, helpers

class_names_list, label_values = helpers.get_label_info(os.path.join(dataset, "class_dict.csv"))
num_classes = len(label_values)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess=tf.Session(config=config)

net_input = tf.placeholder(tf.float32,shape =[None,None,None,3])

with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph('**.meta')
    new_saver.restore(sess,'')
    graph=tf.get_default_graph()
    input_img = graph.get_operation_by_name('input_img').outputs[0]
    output = tf.get_collection("predict")
    cap = cv2.VideoCapture(0)
    frame_index = 0
    while (cap.isOpened()):
        success, frame = cap.read()
        print("---> 正在读取第%d帧:" % frame_index, success)
        if frame_index % interval == 0:
            frame = utils.load_image(frame)
            resize_frame = cv2.resize(frame, (512, 512))
            input_image = np.expand_dims(np.float32(resize_frame, axis=0) / 255.0

            st = time.time()
            output_image = sess.run(output, feed_dict={input_img: input_image})
            run_time = time.time() - st

            output_image = np.array(output_image[0, :, :, :])
            output_image = helpers.reverse_one_hot(output_image)
            out_vis_image = helpers.colour_code_segmentation(output_image, label_values)
            file_name = utils.filepath_to_name(args.image)
            cv2.imwrite("%s_pred.png" % (file_name), cv2.cvtColor(np.uint8(out_vis_image), cv2.COLOR_RGB2BGR))
            print("")
            print("Finished!")
            print("Wrote image " + "%s_pred.png" % (file_name))
    cap.release()
    cv2.destroyAllWindows()
