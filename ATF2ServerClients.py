import requests
import json
import cv2
 
#addr = 'http://10.98.55.177:5000/'

# If testing server, local host address


def imageDifferenceClient(file1, file2, threshold):

  #  payload={'modelName':'imageDifference'}
         
    files = [
        ('file', (file, open(file, 'rb'), 'image/jpg')), ('file2', (file2, open(file2, 'rb'), 'image/jpg')),
        ('modelname',(None, 'imageDifference')), ('threshold',(None, threshold))
    ]
        
    r = requests.post(test_url, files=files)
     
    print("r ", r.content)
    data=r.text
    print(data)
    




server = "local"

if server == "local":
    addr = "http://127.0.0.1:5000"
    
    
test_url = addr + '/image' 

model ="imageDifference" 
file='C:\\Users\\slhahn\\DockerDock\\ScikitImage\\images\\gal1.jpg'
file2 = 'C:\\Users\\slhahn\\DockerDock\\ScikitImage\\images\\round88.jpg'
threshold = "5"
 
 
if model =="imageDifference":
    response = imageDifferenceClient(file, file2, threshold)

 
    
    
    
# #
# # #multi image
# file='C://Users//slhahn//DockerDock//A_DEPLOY//uploads//screensinst.png'
# payload={'modelName':'multiscreens'}
 
 
# files = [
    # ('file', (file, open(file, 'rb'), 'image/jpg')),
    # ('modelname',(None, 'multiscreens')), ('threshold',(None, '0.50')) , ('category',(None, 'instructional')), ('count',(None, '10'))
# ]
 
 # # multi
 
 
# # #Audio
# if file == "audio":
    # file='C://Users//slhahn//DockerDock//A_DEPLOY//uploads//BootAudioFileRecordedOnIteration1.wav'
    # payload={'modelName':'boot'}
     
    # files = [
        # ('file', (file, open(file, 'rb'), 'audio/wav')),
        # ('modelname',(None, 'boot')), ('threshold',(None, '0.50')), ('category',(None, 'instructional')), ('count',(None, '10'))
    # ]
      
  

# # # template

# # file='C://Users//slhahn//DockerDock//A_DEPLOY//uploads//statusBar.jpg'
# # payload={'modelName':'statusbarwheel'}
 
# # files = [
    # # ('file', (file, open(file, 'rb'), 'image/jpg')),
    # # ('modelname',(None, 'statusbarwheel')), ('threshold',(None, '0.50')), ('category',(None, 'instructional')), ('count',(None, '10'))
# # ]
   
  
     
# if file == "image":
# # #change to local image
    # file='C://Users//slhahn//DockerDock//A_DEPLOY//uploads//iconGrid1.jpg'
    # payload={'modelName':'icongrid'}
         
    # files = [
        # ('file', (file, open(file, 'rb'), 'image/jpg')),
        # ('modelname',(None, 'icongrid')), ('threshold',(None, '0.50')) , ('category',(None, 'positive')), ('count',(None, '10'))
    # ] 
  

