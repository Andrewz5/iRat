#import opencv and numpy
from turtle import shape, st
import cv2  
import numpy as np
import imutils
import pickle
import pandas as pd 
from csv import DictWriter
from Rat_Detection import RatDetection
from TrajectoryClasification import TrajectoryClasification,TrajectoryClasificationStrategy,oneDollorRecognize,DTWMethod,FastDTWMethod
from videoProcessing import VideoProcessing
from SaveDataToFile import SaveDataToFile
import os

# def startTracking():
quarterFrame=0.0

videoNumber=1
trajectoryType=[]
confidence=[]
stop=0
flagRatIsIn=1
stopLatancy=1
checkflag=0
latancySec=0
latancyframe=0.0


pointsList=[]
timeFlag=0
programsec=0

cap=cv2.VideoCapture('C1.mp4')#'%i.mp4'%(videoNumber))
cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)


backSub = cv2.createBackgroundSubtractorKNN()

#get the point of each square 
with open('hahahaha', 'rb') as f:
	polygonsWithScore = pickle.load(f)
# print(polygonsWithScore)
print("first cell",polygonsWithScore[0][2])
print("second cell",polygonsWithScore[1][2])
print("third cell",polygonsWithScore[2][2])
print("fourth cell",polygonsWithScore[3][2])

pointtoQuater=''
frameCounter=0
ratDetectionMethods=RatDetection()
frameEdit=VideoProcessing()
trajectoryClassifiction=TrajectoryClasification(FastDTWMethod())
def detectRatMask(img):
	imgBlur=cv2.GaussianBlur(img,(7,7),2)
	kernel = np.ones((5, 5), np.uint8)    
	fgMask = backSub.apply(imgBlur)
	mask=cv2.morphologyEx(fgMask,cv2.MORPH_OPEN,kernel)
	mask = cv2.medianBlur(mask, 9)
	mask = cv2.dilate(mask, kernel, iterations=2)
	return mask
def ROI(frame):
			# Y1  Y2   X1   X2
	roi=frame[50: 550,330: 850]
	return roi

fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('howamendeh.mp4', fourcc, 30.0, (520,490))
while(1):
	frameCounter+=1
	if frameCounter== cap.get(cv2.CAP_PROP_FRAME_COUNT):
		break
	ret,frame=cap.read()
	# frame_c=frame
	if ret:
		# frame=frameEdit.rescaleFrame(frame)
		# print(frame.shape)
		# frame=ROI(frame)
		# print(frame.shape)
		# frame_c=frame
		# ratDetectionMethods.sethsvValue(frame)
		# ratDetectionMethods.setframe(frame)
		# frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  

		fgMask=detectRatMask(frame)
		cnts  = cv2.findContours(fgMask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)   
		cnts = imutils.grab_contours(cnts)
		poly=[]
		if ratDetectionMethods.getCountNumberofEnter()<1:
			latancyframe+=1
		if cap.get(cv2.CAP_PROP_POS_FRAMES)>20:
			if len(cnts) > 0:
					c = max(cnts, key=cv2.contourArea)
					rect = cv2.minAreaRect(c)
					box = cv2.boxPoints(rect)
					box = np.int0(box)
					center=trajectoryClassifiction.determineTheCenter(frame,cnts)
					# print()
					for cr in trajectoryClassifiction.getPoints():
						cv2.circle(frame, cr, 5, (0, 0, 255), -1)
						
					if(trajectoryClassifiction.numberOfPoints()==70):
						Shape,conf=trajectoryClassifiction.trajectoryType()
						
						trajectoryClassifiction.restPoints()
						if (Shape!=None):
							trajectoryType.append(Shape)
							confidence.append(conf)
					# print(x)
					cv2.drawContours(frame,[box],0,(0,0,255),2)
					for polyScore in polygonsWithScore:
						poly = np.array([polyScore[0]], np.int32)
						inside = cv2.pointPolygonTest(poly, center, False)
						# print(inside)
						
						whichQuarter=polyScore[2]
						if inside == 1:
							# print(polyScore[2])
							print(checkflag)
							if (polyScore[2]=='0'):
								checkflag+=1
								whichQuarter=polyScore[2]
							else:
								checkflag=0
							if checkflag==3:
								ratDetectionMethods.setprevFlag(1)
								# prevFlag=1
								ratDetectionMethods.setTargetedQurater(polyScore[2])
								ratDetectionMethods.ratPostion()
							prevpolyID=polyScore[1]
							prevRisk=polyScore[2]
							inside=5

							cv2.drawContours(frame, poly, -1, color=(0, 255, 0))
							if polyScore[2]=='0':
								pointtoQuater='0'
								timeFlag=1

								quarterFrame+=1
							else:
								# stoptime()
								pointtoQuater='1'
								timeFlag=0
		cv2.putText(frame, 'Number of entries %i'%(ratDetectionMethods.getCountNumberofEnter()),(50,400), cv2.FONT_HERSHEY_DUPLEX, 0.5,(0,0,255),2)
		cv2.putText(frame, 'Time inside target quadrant %.2f sec'%((quarterFrame/60)),(50,420), cv2.FONT_HERSHEY_DUPLEX, 0.5,(0,0,255),2)
		cv2.putText(frame, 'Time till reaching target quadrant  %.2f sec'%((latancyframe/60)),(50,440), cv2.FONT_HERSHEY_DUPLEX, 0.5,(0,0,255),2)
		#show image
		cv2.imshow('frame',frame)
		# cv2.imshow('frame_c',frame_c)
		# cv2.imshow('mask',fgMask)
		# cv2.imwrite('Frame.jpg', frame)
		out.write(frame)

		
	else:
		break
	k = cv2.waitKey(1) & 0xFF
	if k == 27:
		break

print("count=============> ",quarterFrame)
print(quarterFrame/60)
print("progromcount=============> ",frameCounter)
print(frameCounter/60)
print("latancy =============> ",latancyframe)
print(latancyframe/60)

data=SaveDataToFile(trajectoryType,confidence,videoNumber)
data.saveTrajactorys()
data.saveData(quarterFrame,ratDetectionMethods.getCountNumberofEnter(),latancyframe)

print('programsec==>%i'%(programsec))
#destroys all window
cap.release()
cv2.destroyAllWindows()