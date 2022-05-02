#import opencv and numpy
from turtle import shape, st
import cv2  
import numpy as np
import imutils
import pickle
import pandas as pd 
import OneDollor_Recognizer_rat
from csv import DictWriter
from Rat_Detection import RatDetection
from videoProcessing import VideoProcessing

quarterFrame=0.0

videoNumber=2
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




cap=cv2.VideoCapture('%i.mp4'%(videoNumber))
cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)

	

backSub = cv2.createBackgroundSubtractorKNN()





#get the point of each square 
with open('polygons4', 'rb') as f:
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
while(1):

	frameCounter+=1
    
	if frameCounter== cap.get(cv2.CAP_PROP_FRAME_COUNT):
		break
	ret,frame=cap.read()
	frame_c=frame
	if ret:
		frame=frameEdit.rescaleFrame(frame)
		frame=frameEdit.ROI(frame)
		frame_c=frame
		ratDetectionMethods.sethsvValue(frame)
		ratDetectionMethods.setframe(frame)
		fgMask=ratDetectionMethods.detectRatMask()
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
					c = max(cnts, key=cv2.contourArea)
					((x, y), radius) = cv2.minEnclosingCircle(c)
					M = cv2.moments(c)
					center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
					cv2.circle(frame, center, 2, (0, 0, 255), -1)
					x,y=center
					pointsList.append(center)
					# for cr in pointsList:
					# 	cv2.circle(frame, cr, 5, (0, 0, 255), -1)
					if(len(pointsList)==70):
						Shape,conf=OneDollor_Recognizer_rat.recognize(pointsList)
						pointsList=[]
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
		cv2.imshow('frame_c',frame_c)
		cv2.imshow('mask',fgMask)
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


dict = {'trajectoryType': trajectoryType, 'confidence': confidence}
df = pd.DataFrame(dict) 
dictData = {'Mouse number': videoNumber, 'Time till reaching ladder (sec)': 0, 
'Time spent inside target quadrant (sec)': '%.2f'%((quarterFrame/60)),'Number of entries': ratDetectionMethods.getCountNumberofEnter()
,'Time till reaching target quadrant (sec)':'%.2f'%((latancyframe/60))}
# saving the dataframe 
df.to_csv('classifications_hsv%i.csv'%(videoNumber)) 
print('programsec==>%i'%(programsec))
field_names = ['Mouse number','Time till reaching ladder (sec)','Time spent inside target quadrant (sec)',
               'Number of entries','Time till reaching target quadrant (sec)']
with open('Morris_water_maze_my_data2 - HSV2.csv','a') as fd:
	writer_object = DictWriter(fd, fieldnames=field_names)
	writer_object.writerow(dictData)
#destroys all window
cv2.destroyAllWindows()