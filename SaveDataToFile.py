
import pandas as pd 
from csv import DictWriter
import time
from datetime import datetime

class SaveDataToFile:
    timestr = time.strftime("%Y%m%D-%H%M%S")
    # print(f"filename_{date}")

    

    def __init__(self,trajectoryType,confidence,videoNumber):
        self.trajectoryType=trajectoryType
        self.confidence=confidence
        self.videoNumber=videoNumber

    def saveTrajactorys(self,):
        date = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
        fileName=f"VideoResult\\Classification\\Morris_water_maze_classifications_{date}.csv"

        dict = {'trajectoryType': self.trajectoryType, 'confidence': self.confidence}
        df = pd.DataFrame(dict) 
        df.to_csv(fileName) 

    def saveData(self,quarterFrame,numberEnters,latancyframe):
        date = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
        fileName=f"VideoResult\\Data\\Morris_water_maze_data_{date}.csv"
        dictData = {'Mouse number': self.videoNumber, 'Time till reaching ladder (sec)': 0, 
        'Time spent inside target quadrant (sec)': '%.2f'%((quarterFrame/60)),'Number of entries': numberEnters
        ,'Time till reaching target quadrant (sec)':'%.2f'%((latancyframe/60))}
        # saving the dataframe 
        
        field_names = ['Mouse number','Time till reaching ladder (sec)','Time spent inside target quadrant (sec)',
                    'Number of entries','Time till reaching target quadrant (sec)']
        with open(fileName,'a') as fd:
            writer_object = DictWriter(fd, fieldnames=field_names)
            writer_object.writerow(dictData)
    
