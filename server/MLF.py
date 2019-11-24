#Machine Learning Framework
import sys
import os
import glob
import re
import numpy as np
import tensorflow as tf
import json
import cv2 as cv
import time
import io
import sys
from PIL import Image
# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

# template match
def opencvMatchIcon(testImage, template, threshold = 0.5, countMatch = 10):

    # prepare new test image for verification, extract crop
    test_image_open=Image.open(testImage)
    test_image=np.array(test_image_open)

    # prepare template icons
    template_image_open=Image.open(template)

    template_image=np.array(template_image_open)

    #get result from opencv template matcher
    result_icon=cv.matchTemplate(test_image, template_image, cv.TM_CCORR_NORMED)

    # construct threshold for icon match to pass
    result_count=np.count_nonzero(result_icon >= threshold)
    print("count of pixels matching template with threshold:", result_count)
    if(result_count>= countMatch):
        print('Found icon.')
        return "Pass"
    else:
        print('Did not find icon.')
        return "Fail"


#Deep Learning Image Model
def model_predict(img_path, model_file,label_file,threshold, category):
  #Disable tensorflow warnings
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
  input_height = 224
  input_width = 224
  input_mean = 128
  input_std = 128
  input_layer = "input"
  output_layer = "final_result"
  resultLine=[]
  category = category.lower()
  graph = load_graph(model_file)
  #Get image tensor
  t = read_tensor_from_image_file(img_path,
                                  input_height=input_height,
                                  input_width=input_width,
                                  input_mean=input_mean,
                                  input_std=input_std)

  input_name = "import/" + input_layer
  output_name = "import/" + output_layer
  input_operation = graph.get_operation_by_name(input_name);
  output_operation = graph.get_operation_by_name(output_name);

  with tf.Session(graph=graph) as sess:

    results = sess.run(output_operation.outputs[0],
                      {input_operation.outputs[0]: t})

  results = np.squeeze(results)

  labels = load_labels(label_file)
  print(results)
  print(labels)
  templatePass = "Test PASS (score={:0.5f})"
  templateFail = "Test FAIL (score={:0.5f})"
  output = None
  
  for i in range(len(labels)):
      newline= str(labels[i]) + "," + str(results[i]) 
      resultLine.append(newline)

      if((labels[i] == category) and (results[i]>=threshold)):
          print(templatePass.format(results[i]))          
          output=templatePass.format(results[i])
          
      elif((labels[i] == category) and (results[i]<threshold)):
          print(templateFail.format(results[i]))          
          output=templateFail.format(results[i])
  
  if resultLine:           
      pred=str(resultLine)
  else:
      pred= "no prediction made"
  
  if not output:
    output = pred
  print("prediction result", pred)
          
  return output
  
def load_graph(model_file):
  """
  Loads the retrained graph
  Args:
    model_file:path to graph file
  Returns:
    graph: tensorflow graph
  """
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)

  return graph
  
def load_labels(label_file):
  """
  Load labels from label file
  Args:
    label_file: path to retrained_labels.txt which contains
                    names of labels for classification
  Returns:
    label: list of labels
  """
  # label = []

  label=label_file.split(";")
  return label
  
  return label

def read_tensor_from_image_file(file_name, input_height=299, input_width=299,
                input_mean=0, input_std=255):
  """
  Convert image into appropriate tensor format before passing to the retrained graph
  Args:
    image_reader: Decoded image from filepath
    input_height: resized image dimensions
    input_weight: resized image dimensions
    input_mean: mean value of image for normalizing
    input_std: std deviation value of input image for normalizing
  Returns:
    result: Normalised image after tensorflow session
  """
  input_name = "file_reader"
  output_name = "normalized"
  file_reader = tf.read_file(file_name, input_name)
  if file_name.endswith(".png"):
    image_reader = tf.image.decode_png(file_reader, channels = 3,
                                       name='png_reader')
  elif file_name.endswith(".gif"):
    image_reader = tf.squeeze(tf.image.decode_gif(file_reader,
                                                  name='gif_reader'))
  elif file_name.endswith(".bmp"):
    image_reader = tf.image.decode_bmp(file_reader, name='bmp_reader')
  else:
    image_reader = tf.image.decode_jpeg(file_reader, channels = 3,
                                        name='jpeg_reader')
  float_caster = tf.cast(image_reader, tf.float32)
  dims_expander = tf.expand_dims(float_caster, 0);
  resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
  normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
  sess = tf.Session()
  result = sess.run(normalized)

  return result  
  
#AUDIO 
  
def AudioPredict(soundfile, class_test,threshold):
    import argparse
    import tensorflow as tf
    import tflearn
    from tflearn.layers.core import input_data, dropout, fully_connected
    from tflearn.layers.conv import conv_2d, max_pool_2d
    from tflearn.layers.normalization import local_response_normalization
    from tflearn.layers.estimator import regression
    import os
    import numpy as np
    import cv2
    from skimage.util.shape import view_as_blocks
    from skimage.util.shape import view_as_windows
    np.random.seed(1001)
    import librosa
    from pytictoc import TicToc
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
    tf.logging.set_verbosity(tf.logging.ERROR)

    patchscore = float(threshold)

    timet = TicToc()
    timet.tic()

    goobeps=0.82
    islandps=0.82
    mlworldps=0.82
    bootps=0.82
    tpcount=2e9
    
    tt=0

    classtestlist=['boot','goobe','island', 'mlworld']
    for p in range(len(classtestlist)):
        if class_test==classtestlist[p]:
            tt+=1
    if tt==0:
    
        print('class_test ', class_test, ' entered is not valid')
        print('acceptable class entries: ', classtestlist)

    print('loading audio file')

    #load sound file, get melspectrogram, crop overlapping tiles and save tensor
    X=np.random.random((1,128,128,1))
    wav1, sr = librosa.core.load(soundfile)
    stepper=int(np.ceil(len(wav1)/sr/6))
    Sxx=librosa.feature.melspectrogram(wav1)
    Sxx2 = np.log10(1+10*abs(Sxx))
    Sxx2norm=255*(Sxx2-Sxx2.min())/(Sxx2.max()-Sxx2.min());
    Sxx2norm=np.round_(Sxx2norm, decimals=0, out=None).astype(int)
    Sxx2norm=Sxx2norm.astype(np.uint8)
 
    if Sxx2norm.shape[1]>=128:
        s1=int(np.floor(Sxx2norm.shape[1]/128)*128)
        B = view_as_windows(Sxx2norm, window_shape=(128, 128),step=stepper) 

        for index in range(B.shape[1]):
            x=B[0][index]
            x = np.expand_dims(x, axis=0)
            x = np.expand_dims(x, axis=3)
            X = np.concatenate((X,x),axis=0)

    # remove random start
    X = X[1:,:,:,:]
    tf.reset_default_graph()

     # number of classes
    y=2
     
     # image preprocessing
    img_prep = tflearn.ImagePreprocessing()
    img_prep.add_featurewise_zero_center()# Zero Center (With mean computed over the whole dataset)
    img_prep.add_featurewise_stdnorm() #STD Normalization (With std computed over the whole dataset)

    # Building 'AlexNet'
    network = input_data(shape=[None, 128, 128, 1],data_preprocessing=img_prep)
    network = conv_2d(network, 96, 11, strides=4, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = conv_2d(network, 256, 5, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = conv_2d(network, 384, 3, activation='relu')
    network = conv_2d(network, 384, 3, activation='relu')
    network = conv_2d(network, 256, 3, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = fully_connected(network, 4096, activation='tanh')
    network = dropout(network, 0.5)
    network = fully_connected(network, 4096, activation='tanh')
    network = dropout(network, 0.5)
    network = fully_connected(network, y, activation='softmax')
    network = regression(network, optimizer='momentum',
                     loss='categorical_crossentropy',
                     learning_rate=0.0001)

    model = tflearn.DNN(network, checkpoint_path='model_alexnet',
                    max_checkpoints=1, tensorboard_verbose=2)

    # set up paramaters for model
    if class_test=='boot':
        classmodel='AudioModels/model_Alexnet_bootnet1/Alexnet_bootnet1' 
        patchscore=bootps
        tilepasscount=np.floor(tpcount/(len(wav1)*len(X)))
        if tilepasscount>len(X):
            tilepasscount=np.floor(0.75*len(X))
        
        print('loading boot test')
    elif class_test=='goobe':
        print('loading goobe test')
        classmodel='AudioModels/model_Alexnet_goobenet1/Alexnet_goobenet1'
        patchscore=goobeps
        tilepasscount=np.floor(tpcount/(len(wav1)*len(X)))
        if tilepasscount>len(X):
            tilepasscount=np.floor(0.25*len(X))

    elif class_test=='island':
        print('loading island test')
        classmodel='AudioModels/model_Alexnet_islandnet1/Alexnet_islandnet1'
        patchscore=islandps
        tilepasscount=np.floor(tpcount/(len(wav1)*len(X)))
        if tilepasscount>len(X):
            tilepasscount=np.floor(0.25*len(X))

    elif class_test=='mlworld':
        print('loading mlworld test')
        classmodel='AudioModels/model_Alexnet_mlworldnet1/Alexnet_mlworldnet1'
        patchscore=mlworldps
        tilepasscount=np.floor(tpcount/(len(wav1)*len(X)))
        if tilepasscount>len(X):
            tilepasscount=np.floor(0.25*len(X))

    
    model.load(classmodel)
    
    print("totaltilecount", len(X))
    print("tilepasscount", tilepasscount)
    lastPF=[]

    # Get prediction of tile list
    res=model.predict(X)
    
    print(res)

    # method 1: PASS if any one tile gets 99% or better
    topmatch=0
    for k in range(len(res[:,0])):
        if res[k,0]>=0.99:
            topmatch+=1
    
    # tally consecutive tiles that meet the threshold
    z = np.where((res[:,0]-patchscore)>0)
    z=z[0]
    tally=[0]
    for j in range(len(z)-1):
        if z[j+1]-z[j]==1:
            tally.append(1)
    if(sum(tally)!=0):
        tally.append(1)
    timet.toc()
    
    if(sum(tally)>=tilepasscount or topmatch>=1):
        lastPF.append(1)
        print(sum(tally),'/',len(X))
    else:

        lastPF.append(0)
        print(sum(tally),'/',len(X))
    
    if np.sum(lastPF)>=1:
   
        print('PASS')
        ret ='PASS'
        return ret
    else:
        print('FAIL')
        ret = 'FAIL'
        return ret
		
def get_circles(image_path, minrad, maxrad):
    image = image = cv.imread(image_path)
    img_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    img = img_gray
    img = cv.medianBlur(img,5)
    dp = 2
    ans = False

    circles = cv.HoughCircles(img,cv.HOUGH_GRADIENT,dp,minDist =100,param1=50,param2=100,minRadius=minrad,maxRadius=maxrad)
    if circles is not None:
        ans = (circles.shape[1]>=1) 
    return ans
        