import cv2
import numpy as np
class RatDetection:
	def __init__(self,img=None,targetedQurater=None,prevFlag=1):
		self.img=img
		self.targetedQurater=targetedQurater
		self.hsv = None
		self.prevFlag=prevFlag
		self.countNumberofEnter=0
		
	def setTargetedQurater(self,targervalue):
		self.targetedQurater=targervalue
	
	def setprevFlag(self,prevFlag):
		self.prevFlag=prevFlag
	
	def setCountNumberofEnter(self,countNumberofEnter):
		self.countNumberofEnter=countNumberofEnter
	
	def getCountNumberofEnter(self):
		return self.countNumberofEnter

	def setframe(self,frame):
		self.img=frame	
	def sethsvValue(self,frame):
		self.hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

	
	# =====================================
	# Image subtration and blur 
	# =====================================
	def detectRatMask(self):
		imgBlur=cv2.GaussianBlur(self.img,(7,7),2)
		LowerRegion=np.array([0,0,76] ,np.uint8)
		upperRegion = np.array( [ 40,137,255]   ,np.uint8)
		redObject = cv2.inRange(self.hsv,LowerRegion,upperRegion)
		kernel = np.ones((5, 5), np.uint8)    
		# fgMask = backSub.apply(imgBlur)
		mask=cv2.morphologyEx(redObject,cv2.MORPH_OPEN,kernel)
		mask = cv2.medianBlur(mask, 9)
		mask = cv2.dilate(redObject, kernel, iterations=2)
		return mask


	# =====================================
	# Number of enters of the rat
	# =====================================
	def ratPostion(self):
		global flagRatIsIn,frame
		if(self.targetedQurater=='0'):
			if(self.prevFlag==1):
				self.countNumberofEnter+=1
				print(self.countNumberofEnter)
				self.prevFlag=0
		else:
			self.prevFlag=1

