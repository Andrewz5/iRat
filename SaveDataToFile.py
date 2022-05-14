
import pandas as pd 
from csv import DictWriter
class SaveDataToFile:
    def __init__(self,trajectoryType,confidence,videoNumber):
        self.trajectoryType=trajectoryType
        self.confidence=confidence
        self.videoNumber=videoNumber

    def saveTrajactorys(self,):
        dict = {'trajectoryType': self.trajectoryType, 'confidence': self.confidence}
        df = pd.DataFrame(dict) 
        df.to_csv('classifications_hsv%i.csv'%(self.videoNumber)) 

    def saveData(self,quarterFrame,numberEnters,latancyframe):
        dictData = {'Mouse number': self.videoNumber, 'Time till reaching ladder (sec)': 0, 
        'Time spent inside target quadrant (sec)': '%.2f'%((quarterFrame/60)),'Number of entries': numberEnters
        ,'Time till reaching target quadrant (sec)':'%.2f'%((latancyframe/60))}
        # saving the dataframe 
        
        field_names = ['Mouse number','Time till reaching ladder (sec)','Time spent inside target quadrant (sec)',
                    'Number of entries','Time till reaching target quadrant (sec)']
        with open('Morris_water_maze_my_data2 - HSV2.csv','a') as fd:
            writer_object = DictWriter(fd, fieldnames=field_names)
            writer_object.writerow(dictData)
    
