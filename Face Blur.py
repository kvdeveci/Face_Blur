import  cv2 as cv
from cvzone.FaceDetectionModule import FaceDetector
cap = cv.VideoCapture(0)
dection =  FaceDetector(minDetectionCon=0.75)
while True:
    ret,frame  =  cap.read()
    frame,bboxs = dection.findFaces(frame,draw=True)
    if bboxs:
        for i,bbox in enumerate(bboxs):
            x,y,w,h = bbox['bbox']
            if x < 0: x = 0
            if y < 0: y = 0
            imCrop = frame[y:y+h,x:x+w]
          
            imgBlur  = cv.blur(imCrop,(50,50))
            frame[y:y+h ,x:x+w] = imgBlur
    cv.imshow("frame",frame)    

    cv.waitKey(1)