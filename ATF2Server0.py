from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from ATF2MLtools import AudioProcessing, AudioDeepLearning, ImageDeepLearning
from ATF2ImageProcessing import ImageProcessing
import datetime

from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
from werkzeug.serving import run_with_reloader
from werkzeug.debug import DebuggedApplication

# Define a flask app
app = Flask(__name__)

modelRootDir = 'models'
print("Running ATF2 Machine Learning Server at", datetime.datetime.now())


@app.route('/image', methods=['GET', 'POST']) 
def upload():
    basepath = os.path.dirname(__file__)
    if request.method == 'POST':        
        pred = "null"
        mod = ImageProcessing()

        f1 = request.files['file']
        f2 = request.files['file2']
        
        posted_file1 = os.path.join(basepath, 'server\\uploads\\ImageProcessing', secure_filename(f1.filename))
        posted_file2 = os.path.join(basepath, 'server\\uploads\\ImageProcessing',secure_filename(f2.filename))
        
        posted_threshold = request.form['threshold']      
                
        f1.save(posted_file1)
        f2.save(posted_file2)
        print(posted_file1)
        print(posted_file2)
        pred = mod.GetDiffTwoImages(posted_file1, posted_file2, posted_threshold)
        
        return pred
           

   


if __name__ == '__main__':

    http_server = WSGIServer(('0.0.0.0', 5000), DebuggedApplication(app))
    http_server.serve_forever()



