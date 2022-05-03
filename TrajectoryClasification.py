import string
import cv2
from dollarpy import Recognizer, Template, Point
from abc import ABC, abstractmethod
from dtaidistance import dtw_ndim
import numpy as np
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
class TrajectoryClasificationStrategy(ABC):
    
    @abstractmethod
    def startTrajectoy(self,points):
        pass
    


class oneDollorRecognize(TrajectoryClasificationStrategy):
    def startTrajectoy(self,points):
        z=[]

        tmpl_1 = Template('incursion',[Point(175, 359), Point(177, 379), Point(184, 395), Point(197, 398), Point(222, 395), Point(238, 381), Point(258, 360), Point(278, 335), Point(301, 305), Point(323, 279), Point(336, 256), Point(354, 237), Point(361, 226), Point(386, 188), Point(392, 177), Point(400, 169), Point(405, 160)])
        tmpl_2 = Template('scaning',[Point(585, 98), Point(576, 104), Point(572, 113), Point(557, 127), Point(542, 140), Point(528, 156), Point(501, 182), Point(482, 198), Point(455, 215), Point(448, 216), Point(431, 213), Point(432, 208), Point(441, 178), Point(449, 161), Point(470, 139), Point(484, 117), Point(494, 110), Point(506, 96), Point(520, 87), Point(532, 78), Point(537, 61)])
        tmpl_3 = Template('focuedSearch',[Point(494, 187), Point(487, 219), Point(449, 237), Point(453, 186), Point(506, 186), Point(496, 259), Point(473, 213), Point(481, 163), Point(542, 248), Point(472, 254), Point(436, 184), Point(534, 184), Point(462, 224)])
        tmpl_4 = Template('chainingResponse',[Point(88, 232), Point(106, 276), Point(123, 295), Point(132, 313), Point(148, 328), Point(179, 340), Point(189, 339), Point(224, 337), Point(246, 332), Point(267, 322), Point(279, 306), Point(288, 289), Point(297, 271), Point(304, 248), Point(311, 223), Point(315, 211)])
        tmpl_5=Template('chainingResponse',[Point(511, 270), Point(477, 287), Point(441, 300), Point(414, 300), Point(327, 284), Point(311, 254), Point(309, 223), Point(309, 210)])

        tmpl_6 = Template('selfOrienting',[Point(565, 165), Point(556, 185), Point(540, 193), Point(509, 193), Point(471, 193), Point(449, 171), Point(453, 145), Point(473, 127), Point(484, 144), Point(490, 177), Point(488, 197), Point(483, 212), Point(473, 225), Point(440, 236), Point(417, 244)])
        tmpl_7 = Template('incursion',[Point(329, 81), Point(308, 67), Point(278, 51), Point(253, 72), Point(254, 101), Point(258, 135), Point(255, 162), Point(257, 202), Point(257, 237), Point(256, 270), Point(251, 296)])
        tmpl_8 = Template('selfOrienting',[Point(548, 173), Point(536, 183), Point(518, 186), Point(500, 181), Point(495, 161), Point(503, 170), Point(503, 174), Point(500, 190), Point(485, 232)])    
        recognizer = Recognizer([tmpl_1,tmpl_2,tmpl_3,tmpl_4,tmpl_5,tmpl_6,tmpl_7,tmpl_8])
        for pt in points:
            x,y=pt
            z.append(Point(x,y))
        # Call 'recognize(...)' to match a list of 'Point' elements to the previously defined templates.
        result = recognizer.recognize(z)
        shape,conf=result
        if (shape!=None):
            print("1$ ==> ",result)  # Output: ('X', 0.733770116545184)

        return result

class DTWMethod(TrajectoryClasificationStrategy):
        def startTrajectoy(self,points):
            ratPaths=["incursion","scaning","focuedSearch","chainingResponse","selfOrienting"]
            lspaths=[]
            distLs=[]
            incursion = np.array([[238, 309], [268, 341], [308, 332], [344, 292], [390, 217], [413, 167], [455, 112], [480, 80], [490, 69]])

            scaning = np.array([[590, 73], [577, 95], [556, 132], [522, 176], [505, 206], [466, 252], [435, 285], [395, 283], [396, 259], [415, 186],[442, 140], [459, 117], [461, 87], [457, 54]])
            focuedSearch = np.array([[449, 280], [457, 295], [430, 291], [450, 276], [486, 289], [436, 317], [431, 276], [502, 266], [467, 306], [399, 267], [451, 261], [491, 319], [389, 305], [475, 256]])
            chainingResponse = np.array([[141, 158], [144, 197], [151, 237], [152, 263], [160, 292], [172, 321], [188, 341], [206, 348], [246, 353], [283, 355], [323, 351], [344, 330], [353, 308], [365, 279], [368, 242], [365, 209], [366, 175], [366, 159]])
            selfOrienting = np.array([[502, 238], [486, 256], [456, 266], [429, 262], [424, 243], [435, 228], [452, 241], [455, 252], [455, 267], [455, 282], [451, 305]])
            lspaths.append(incursion)
            lspaths.append(scaning)
            lspaths.append(focuedSearch)
            lspaths.append(chainingResponse)
            lspaths.append(selfOrienting)
            
            # print(lspaths)
            for oneMove in lspaths:
                distance = dtw_ndim.distance(oneMove, points)
                # print("Today ==>",distance)
                # distance, path = fastdtw(oneMove, points, dist=euclidean)
                distLs.append(distance)

            mindis=min(distLs)
            print("DTW ==> ",ratPaths[distLs.index(mindis)],"==>",mindis)
            # print(path)
            return ratPaths[distLs.index(mindis)],mindis


class FastDTWMethod(TrajectoryClasificationStrategy):
        def startTrajectoy(self,points):
            ratPaths=["incursion","scaning","focuedSearch","chainingResponse","selfOrienting"]
            lspaths=[]
            distLs=[]
            incursion = np.array([[238, 309], [268, 341], [308, 332], [344, 292], [390, 217], [413, 167], [455, 112], [480, 80], [490, 69]])
            scaning = np.array([[590, 73], [577, 95], [556, 132], [522, 176], [505, 206], [466, 252], [435, 285], [395, 283], [396, 259], [415, 186],[442, 140], [459, 117], [461, 87], [457, 54]])
            focuedSearch = np.array([[449, 280], [457, 295], [430, 291], [450, 276], [486, 289], [436, 317], [431, 276], [502, 266], [467, 306], [399, 267], [451, 261], [491, 319], [389, 305], [475, 256]])
            chainingResponse = np.array([[141, 158], [144, 197], [151, 237], [152, 263], [160, 292], [172, 321], [188, 341], [206, 348], [246, 353], [283, 355], [323, 351], [344, 330], [353, 308], [365, 279], [368, 242], [365, 209], [366, 175], [366, 159]])
            selfOrienting = np.array([[502, 238], [486, 256], [456, 266], [429, 262], [424, 243], [435, 228], [452, 241], [455, 252], [455, 267], [455, 282], [451, 305]])
            lspaths.append(incursion)
            lspaths.append(scaning)
            lspaths.append(focuedSearch)
            lspaths.append(chainingResponse)
            lspaths.append(selfOrienting)

            # print(lspaths)
            for oneMove in lspaths:
                distance, path = fastdtw(oneMove, points, dist=euclidean)
                distLs.append(distance)

            mindis=min(distLs)
            print("FastDTW ==> ",ratPaths[distLs.index(mindis)],"==>",mindis)
            # print(path)
            return ratPaths[distLs.index(mindis)],mindis


class TrajectoryClasification:
    def __init__(self, trajectoryStrategyType: TrajectoryClasificationStrategy):
        self.pointsList=[]
        self.confidence=[]
        self.trajectoryStrategyType=trajectoryStrategyType
    
    def trajectoryType(self):
        chossenTypeResuts = self.trajectoryStrategyType.startTrajectoy(self.pointsList)

        return chossenTypeResuts


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