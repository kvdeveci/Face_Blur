import  cv2 as cv
import cvzone 
from cvzone.FaceDetectionModule import FaceDetector#facedetect
cap = cv.VideoCapture(0)
dection =  FaceDetector(minDetectionCon=0.75)#facedetect
while True:
    ret,frame  =  cap.read()
    frame,bboxs = dection.findFaces(frame,draw=True)#facedetect
    if bboxs:
        for i,bbox in enumerate(bboxs):
            x,y,w,h = bbox['bbox']
            if x < 0: x = 0
            if y < 0: y = 0
            imCrop = frame[y:y+h,x:x+w]#it gives start and end point 
           # cv.imshow(f'cropped :{i}',imCrop)
            imgBlur  = cv.blur(imCrop,(50,50))
            frame[y:y+h ,x:x+w] = imgBlur
    cv.imshow("frame",frame)    

    cv.waitKey(1)