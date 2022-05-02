import cv2
class VideoProcessing:
    def __init__(self,frame=None,scale=0.25):
        self.frame=frame
        self.scale= scale

    def rescaleFrame(self,frame):
        width=int(frame.shape[1]*self.scale)
        height=int(frame.shape[0]*self.scale)
        dimensions=(width,height)
        return cv2.resize(frame,dimensions,interpolation=cv2.INTER_AREA)
    ## Now I will make it hardcoded
    def ROI(self,frame):
               # Y1  Y2   X1   X2
        roi=frame[50: 500,330: 800]
        return roi
