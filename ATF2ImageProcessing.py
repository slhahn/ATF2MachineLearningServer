import cv2 as cv
import numpy as np
from sklearn.metrics.pairwise import paired_distances
     
# Image Processing Class that calls scikit-learn and opencv methods
class ImageProcessing():

    # If the answer is 0, the two images are exactly the same
    def GetDiffTwoImages(self,fileA, fileB, threshold):  
            ans = "null"
            imgA3 = cv.imread(fileA) # read image
            X = cv.cvtColor(imgA3, cv.COLOR_BGR2GRAY) # color to grayscale

            imgB3=cv.imread(fileB)
            Y = cv.cvtColor(imgB3, cv.COLOR_BGR2GRAY)
            
            if X.shape == Y.shape: 
                d_manhattan=paired_distances(X,Y, metric='manhattan')
                d_man_norm  = np.float(sum(d_manhattan)/(X.shape[0]*X.shape[1]))
            
                threshold = np.float(threshold)
                check = (d_man_norm > threshold)

                if check:
                    ans = "fail"
                    
                checkPass = (d_man_norm <= threshold)

                if checkPass:
                    ans = "pass"            
                
                stringResponse = ans + " " + str(d_man_norm)
                return stringResponse
                
            else:
                ans="error: images not same size"
            
            return ans
            
    def MatchTemplateInImage(self, templatePath, imagePath, minDist, threshold):
        template3 = cv2.imread(templatePath)
        template =cv2.cvtColor(template3, cv2.COLOR_BGR2GRAY)

        image3 = cv2.imread(imagePath)
        image =cv2.cvtColor(image3, cv2.COLOR_BGR2GRAY)
        result = match_template(image, template, pad_input = True)
        peaks = peak_local_max(result,min_distance=minDist,threshold_rel= threshold)
        
        return peaks

                   
    def get_circles(self, image_path, minrad, maxrad):
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
            


    # template match
    def opencvMatchSmallTemplate(self, testImage, template, threshold = 0.5, countMatch = 10):

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
