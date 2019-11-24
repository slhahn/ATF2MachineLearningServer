import requests
import json
import cv2
 
addr = 'http://10.98.55.177:5000/'

# If testing server, local host address
#addr = "http://127.0.0.1:5000"
test_url = addr + '/predict'
test_url2 = addr + '/scan'

# #change to local image
#file='C://Users//slhahn//DockerDock//A_DEPLOY//uploads//iconGrid1.jpg'
# payload={'modelName':'icongrid'}
 
 
# files = [
    # ('file', (file, open(file, 'rb'), 'image/jpg')),
    # ('modelname',(None, 'icongrid')), ('threshold',(None, '0.50')) , ('category',(None, 'positive')), ('count',(None, '10'))
# ]
 

# #
# # #multi image
# file='C://Users//slhahn//DockerDock//A_DEPLOY//uploads//screensinst.png'
# payload={'modelName':'multiscreens'}
 
 
# files = [
    # ('file', (file, open(file, 'rb'), 'image/jpg')),
    # ('modelname',(None, 'multiscreens')), ('threshold',(None, '0.50')) , ('category',(None, 'instructional')), ('count',(None, '10'))
# ]
 
 # # multi
 

 # #circles
file='C:\\Users\\slhahn\\DockerDock\\ScikitImage\\iconSmall.bmp'
payload={'modelName':'circles'}
 
files = [
    ('file', (file, open(file, 'rb'), 'img/bmp')),
    ('modelname',(None, 'circles')), ('category',(None, 'positive')), ('threshold',(None, '250;550'))
]
  
 
 
# # #Audio
# file='C://Users//slhahn//DockerDock//A_DEPLOY//uploads//BootAudioFileRecordedOnIteration1.wav'
# payload={'modelName':'boot'}
 
# files = [
    # ('file', (file, open(file, 'rb'), 'audio/wav')),
    # ('modelname',(None, 'boot')), ('threshold',(None, '0.50')), ('category',(None, 'instructional')), ('count',(None, '10'))
# ]
  
  

# # template

# file='C://Users//slhahn//DockerDock//A_DEPLOY//uploads//statusBar.jpg'
# payload={'modelName':'statusbarwheel'}
 
# files = [
    # ('file', (file, open(file, 'rb'), 'image/jpg')),
    # ('modelname',(None, 'statusbarwheel')), ('threshold',(None, '0.50')), ('category',(None, 'instructional')), ('count',(None, '10'))
# ]
   
  
  
  
r = requests.post(test_url, files=files)
 
print("r ", r.content)
data=r.text
print(data)

