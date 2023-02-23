import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
 
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from grabscreen import grab_screen
import cv2
from IPython import get_ipython


 
sys.path.append("..")
 

 
from utils import label_map_util
from utils import visualization_utils as vis_util


 
MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
 

PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

 
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
 
NUM_CLASSES = 90
 



opener = urllib.request.URLopener()
opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
tar_file = tarfile.open(MODEL_FILE)
for file in tar_file.getmembers():
  file_name = os.path.basename(file.name)
  if 'frozen_inference_graph.pb' in file_name:
    tar_file.extract(file, os.getcwd())
 

 
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()

    
    with tf.compat.v2.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
 

 
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


 
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im.height, im_width, 3)).astype(np.uint8)
 
 
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('pygtaVideotest20220303-1.avi', fourcc, 8.0 , (800,600))


with detection_graph.as_default():
    with tf.compat.v1.Session(graph=detection_graph) as sess:
        while True:
            screen = cv2.resize(grab_screen(region=(0,40,800,600)), (800,600))
            image_np = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)

 
            image_np_expanded = np.expand_dims(image_np, axis=0)
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
 
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
 
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
 
            (boxes, scores, classes, num_detections) = sess.run([boxes, scores, classes, num_detections], feed_dict={image_tensor: image_np_expanded})

 
            vis_util.visualize_boxes_and_labels_on_image_array(
                    image_np,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=8)

            for i, b in enumerate(boxes[0]):
                if classes[0][i] == 3 or classes[0][i] == 6 or classes[0][i] == 8:
                    if scores[0][i] > 0.5:
                        mid_x = (boxes[0][i][3] + boxes[0][i][1]) / 2
                        mid_y = (boxes[0][i][2] + boxes[0][i][0]) / 2
                        apx_distance = round((1-(boxes[0][i][3] - boxes[0][i][1]))**4, 1)
                        cv2.putText(image_np, '{}'.format(apx_distance), (int(mid_x*800), int(mid_y*600)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                        if apx_distance <= 0.5:
                            if mid_x > 0.3 and mid_x < 0.7:
                                cv2.putText(image_np, 'WARNING!!', (int(mid_x*800)-50, int(mid_y*600)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
                        
                            
 
            cv2.imshow('gta-objectdetection', image_np)
            out.write(image_np)

            
            if cv2.waitKey(25) & 0xff == ord('q'):
                cv2.destroyAllWindows()
                break
