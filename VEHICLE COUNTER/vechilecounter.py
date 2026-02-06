import cv2 #Opencv library
import numpy as np #An array object of arbitrary homogeneous items


# Web camera start
cap = cv2.VideoCapture('inputvideo.mp4')

min_width_rect=80   #min width Rectangle
min_height_rect=80   #min height Rectangle

count_line_position= 550
#Initialize Substructor
algo= cv2.bgsegm.createBackgroundSubtractorMOG() #To detect vehicle

def center_handle(x,y,w,h):
    x1=int(w/2)
    y1=int(h/2)
    cx = x+x1
    cy = y+y1
    return cx,cy

detect= []
offset=6  #Allowable error between pixel
counter=0

while True:
    ret,frame1=cap.read()
    grey = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY) 
    #to convert it into grey
    blur = cv2.GaussianBlur(grey,(3,3),5)
    #applying on each frame
    img_sub = algo.apply(blur)
    #dilate is used to extract image out by pixelating it
    dialt = cv2.dilate(img_sub,np.ones((5,5))) 
    # getstructuring allow the algo to get the shape of image
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    # to provide the shape to multichannel images into black and white console
    dilatada = cv2.morphologyEx(dialt,cv2.MORPH_CLOSE, kernel) 
    dilatada = cv2.morphologyEx(dilatada,cv2.MORPH_CLOSE, kernel)
    counterShape,h = cv2.findContours(dilatada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #to decrease size of image 

    cv2.line(frame1,(25,count_line_position),(1200,count_line_position),(255,125,0),3) #setup line positioning
    
    #loop is because of multiple no. of vehicle
    for (i,c) in enumerate(counterShape): 
        (x,y,w,h) = cv2.boundingRect(c)
        validate_counter = (w>=min_width_rect) and (h>=min_height_rect) # validating the width and height of rectangle 
        if not validate_counter:
            continue

        cv2.rectangle(frame1,(x,y),(x+w,y+h),(0,255,0),2)        
    
        center = center_handle(x,y,w,h) #pointiing a circle when a vehicle is crossing line
        detect.append(center)
        cv2.circle(frame1,center,4,(0,0,255),-1)


        for(x,y) in detect:
            if y<(count_line_position+offset) and y>(count_line_position-offset):  
                counter+=1
              #line color changes while vehicle is crossing
            cv2.line(frame1,(25,count_line_position),(1200,count_line_position),(0,125,255),3) 
            detect.remove((x,y))
            
            print("Vehicle Counter:"+str(counter))

    cv2.putText(frame1,"VEHICLE COUNTER :"+str(counter),(450,70),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),5)        


 # cv2.imshow('Detector',dilatada)
    cv2.imshow('Video Original',frame1)

    if cv2.waitKey(1) == 13: #To close window after Enter
        break

cv2.destroyAllWindows()
cap.release()    


