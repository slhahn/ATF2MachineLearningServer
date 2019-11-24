#from __future__ import division, print_function
#Import Libraries
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import tensorflow as tf
import json
from MLF import model_predict, AudioPredict, opencvMatchIcon, get_circles

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
from werkzeug.serving import run_with_reloader
from werkzeug.debug import DebuggedApplication

# Define a flask app
app = Flask(__name__)

modelRootDir = 'models'
print("go to local host")

ModelList = ['bootanimation:retrained_graph_bootanim_batfix1.pb:negative;positive', 'browser:retrained_graph_browser.pb:negative;positive',  'create:retrained_graph_createapp.pb:negative;positive','icongrid:retrained_graph_icon_grid.pb:negative;positive', 'drg:retrained_graph_DrG.pb:negative;positive',  'mlwelcomescreen:retrained_graph_mlWelcomeScreen2.pb:negative;positive',  'screens:retrained_graph_screens.pb:negative;positive',  'multiscreens:retrained_graph_ScreensMulti.pb:instructional;none;wallpaper;whales','gallery:retrained_graph_gallery.pb:negative;positive','boot::audio','goobe::audio','island::audio', 'statusbarwheel:statusbar_template_wheel.bmp:template', 'statusbarwifi:statusbar_template_wifi.bmp:template', 'heyluminmicrophone:microphone_template_heylumin.bmp:template', "circles:circles:circles"]
  

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html', data=[{'name':'boot_animation'}, {'name':'screens_application'}, {'name':'icon_grid'}])

       
@app.route("/test" , methods=['GET', 'POST'])       
def test():
    select = request.form.get('comp_select')
    print("select: ", select)
    modelname=str(select)
    return str(select)


@app.route('/predict', methods=['GET', 'POST'])  
def upload():
    if request.method == 'POST':

        posted_file = request.files['file']
        assert isinstance(posted_file.filename, object)
        
        print("posted_file: ", posted_file.filename)                                          
        posted_model = request.form['modelname'].lower()
        print('posted_model: ', posted_model)
        
        posted_threshold = request.form['threshold']
        print('posted_threshold: ', posted_model)
        posted_category = request.form['category'].lower()
     
        print("model: ", posted_model)  
        print("threshold: ", posted_threshold)
        print("category: ", posted_category)
               
        
        f = request.files['file']       
        print("f: ", f)
        print("filename: ", f.filename)
        
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        posted_model = posted_model.lower()
                      
        for i in range(len(ModelList)):
            paramlist = ModelList[i].split(":")
            
            if posted_model == paramlist[0]:
                print("posted_model found in list, paramlist[0] :", paramlist[0])
                class_model = paramlist[1]
                class_label = paramlist[2]
                print(paramlist[0])
                print(paramlist[1])
                print(paramlist[2])
                if "audio" in  paramlist[2]:
                    posted_threshold = float(posted_threshold)
                    class_model = paramlist[0]
                    preds = AudioPredict(file_path, class_model, posted_threshold)
                    print("audio predictions:", preds)
                    return preds
 
                elif "template" in paramlist[2]:
                    posted_threshold = float(posted_threshold)                
                    posted_count = float(request.form['count'])                 
                    template = paramlist[1]             
                    pred = opencvMatchIcon(file_path, template, posted_threshold, posted_count)
                    
                    return pred
                    
                elif "circles" in paramlist[2]:
                    minrad = int(request.form['threshold'].split(";")[0])
                    maxrad = int(request.form['threshold'].split(";")[1])
                    preds = get_circles(file_path, minrad, maxrad)
                    return str(preds)
                
                # standard image inference
                else: 
                    posted_threshold = float(posted_threshold)                
                    class_model_path = os.path.join(modelRootDir, class_model)
                    print("class_model_path:", class_model_path)
                  # Make prediction
                    preds = model_predict(file_path, class_model_path, class_label, posted_threshold, posted_category)
                    print(preds)

                    return preds

    return "post received, but no prediction returned"


if __name__ == '__main__':

    #http_server = WSGIServer(host='0.0.0.0',debug=True)
    http_server = WSGIServer(('0.0.0.0', 5000), DebuggedApplication(app))
    http_server.serve_forever()
    #app.run(host='0.0.0.0')


