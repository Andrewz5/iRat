import cv2
class TrajectoryClasification:
    def __init__(self):
        self.pointsList=[]
        self.trajectoryType=[]
        self.confidence=[]
    
    def getPoints(self):
        return self.pointsList
    
    def restPoints(self):
        self.pointsList =[]
    
    def numberOfPoints(self):
        return len(self.pointsList)

    def determineTheCenter(self,frame,cnts):
        c = max(cnts, key=cv2.contourArea)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        cv2.circle(frame, center, 2, (0, 0, 255), -1)
        self.pointsList.append(center)
        return center

    # def 