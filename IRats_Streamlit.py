import streamlit as st
import moviepy.editor as moviepy
import cv2
import pickle
import numpy as np
import imutils
from Rat_Detection import RatDetection
from TrajectoryClasification import TrajectoryClasification,TrajectoryClasificationStrategy,oneDollorRecognize,DTWMethod,FastDTWMethod
from videoProcessing import VideoProcessing
from SaveDataToFile import SaveDataToFile
import pyrebase
from streamlit import caching
import time
import base64
timestr = time.strftime("%Y%m%D-%H%M%S")

firebaseConfig = {
    'apiKey': "AIzaSyB-1OSqlNCPIaPVnrZc9iFFvn9f7STxkFY",
    'authDomain': "irats-530cd.firebaseapp.com",
    'projectId': "irats-530cd",
    'databaseURL': "https://irats-530cd-default-rtdb.firebaseio.com/users",
    'storageBucket': "irats-530cd.appspot.com",
    'messagingSenderId': "150724454620",
    'appId': "1:150724454620:web:c608246cfce551910cac19"
  };

# FireBase Authentication
firebase=pyrebase.initialize_app(firebaseConfig)
auth=firebase.auth()

db=firebase.database()
# Firebase Storage
storage=firebase.storage()
logedin=False
SelectedUser=""
resultList=[]
openFieldResults=[]
#============================================================================== 
#== Streamlit Start
# =============================================================================
headerSection = st.container()
mainSection = st.container()
loginSection = st.container()
logOutSection = st.container()

def MorrisWaterMaze(video):
    global resultList
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
    ret, frame = cap.read()
    h, w = frame.shape[:2]
    backSub = cv2.createBackgroundSubtractorKNN()
    with open('polygons4', 'rb') as f:
            polygonsWithScore = pickle.load(f)
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
    htmlvar=cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter('detected_video.mp4', fourcc, 30.0, (520,490))
    trajectoryType=[]
    confidence=[]
    latancyframe=0.0
    checkflag=0
    quarterFrame=0.0

    while(1):
        frameCounter+=1
        if frameCounter== cap.get(cv2.CAP_PROP_FRAME_COUNT):
            break
        ret,frame=cap.read()
        # frame_c=frame
        if ret:
            frame=frameEdit.rescaleFrame(frame)
            # print(frame.shape)
            frame=ROI(frame)
            # print(frame.shape)
            frame_c=frame
            # ratDetectionMethods.sethsvValue(frame)
            # ratDetectionMethods.setframe(frame)
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
                                # print(checkflag)
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
            # TimeSpentQuadrant= 
            cv2.putText(frame, 'Number of entries %i'%(ratDetectionMethods.getCountNumberofEnter()),(50,400), cv2.FONT_HERSHEY_DUPLEX, 0.5,(0,0,255),2)
            cv2.putText(frame, 'Time inside target quadrant %.2f sec'%((quarterFrame/60)),(50,420), cv2.FONT_HERSHEY_DUPLEX, 0.5,(0,0,255),2)
            cv2.putText(frame, 'Time till reaching target quadrant  %.2f sec'%((latancyframe/60)),(50,440), cv2.FONT_HERSHEY_DUPLEX, 0.5,(0,0,255),2)
            # show image
            cv2.imshow('frame',frame)
            cv2.imshow('frame_c',frame_c)
            cv2.imshow('mask',fgMask)
           
            
            out.write(frame)
    

        else:

            break

    print("count=============> ",quarterFrame)
    print(quarterFrame/60)
    print("progromcount=============> ",frameCounter)
    print(frameCounter/60)
    print("latancy =============> ",latancyframe)
    print(latancyframe/60)

    data=SaveDataToFile(trajectoryType,confidence,2)
    data.saveTrajactorys()
    data.saveData(quarterFrame,ratDetectionMethods.getCountNumberofEnter(),latancyframe)

    data={"Number of entires": ratDetectionMethods.getCountNumberofEnter(),"Time inside target quadrant": "{:.2f}".format(quarterFrame/60),"Time till reaching target quadrant":"{:.2f}".format(latancyframe/60)}
    resultList.append(data)

def OpenField(video):
    OpenFieldsec=0
    min=0
    hr=0
    global openFieldResults


    l_b=np.array([144,9,78])# lower hsv bound for red
    u_b=np.array([144,5,80])# upper hsv bound to red

    backSub = cv2.createBackgroundSubtractorKNN()
    object_detector=cv2.createBackgroundSubtractorMOG2(history=100,varThreshold=60)

    cap= cv2.VideoCapture(video)
    frameCounter=0
    count=0
    riskcount=0
    #71,201 764,182 87,873 762,863
    corrnerPoints=[[191, 110], [576, 107], [176, 482], [565, 508]]

    #To get the point of the corrner board
    def getBoard2(img):
        width, height= int(500*1.5), int(350*1.5)
        pts1=np.float32(corrnerPoints)
        pts2=np.float32([[0,0],[width,0],[0,height],[width,height]])
        matrix=cv2.getPerspectiveTransform(pts1,pts2)
        imgOutput=cv2.warpPerspective(img,matrix,(width,height))
        for x in range(4):
            cv2.circle(img,(corrnerPoints[x][0],corrnerPoints[x][1]),5,(0,255,0),cv2.FILLED)
        return imgOutput
    #get the point of each square 
    with open('tyr', 'rb') as f:
        polygonsWithScore = pickle.load(f)
    # print(polygonsWithScore)

    # Image subtration and blur 
    def detectRatMask(img):
        imgBlur=cv2.GaussianBlur(img,(7,7),2)
        kernel = np.ones((5, 5), np.uint8)    
        fgMask = backSub.apply(imgBlur)
        mask=cv2.morphologyEx(fgMask,cv2.MORPH_OPEN,kernel)
        mask = cv2.medianBlur(mask, 9)
        mask = cv2.dilate(mask, kernel, iterations=5)
        return mask
    prevpolyID=-1

    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter('detected_video.mp4', fourcc, 30.0, (750,525))
    while True:
        
        frameCounter+=1
        
        if frameCounter== cap.get(cv2.CAP_PROP_FRAME_COUNT):
            break

        ret, frame = cap.read()
        # if(frameCounter>800):
        hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        HSV_mask=cv2.inRange(hsv,l_b,u_b)
        mask=object_detector.apply(frame)

        if (frameCounter%20==0):
                imgBoard= getBoard2(frame)
                fgMask=detectRatMask(imgBoard)
                cnts  = cv2.findContours(fgMask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)   
                cnts = imutils.grab_contours(cnts)
                # center = None
                _,mask=cv2.threshold(mask,254,255,cv2.THRESH_BINARY)
                _,HSV_mask=cv2.threshold(HSV_mask,254,255,cv2.THRESH_BINARY)
                
                # only proceed if at least one contour was found
                if len(cnts) > 0:
                    c = max(cnts, key=cv2.contourArea)
                    rect = cv2.minAreaRect(c)
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    cv2.drawContours(imgBoard,[box],0,(255,0,0),2)
                    x,y,w,h=cv2.boundingRect(box)
                    c = max(cnts, key=cv2.contourArea)
                    ((x, y), radius) = cv2.minEnclosingCircle(c)
                    M = cv2.moments(c)
                    center = (int(M["m10"] / M["m00"])+40, int(M["m01"] / M["m00"]))
                    cv2.circle(imgBoard, center, 5, (0, 0, 255), -1)
                    flag=0
                    # print(center)
                    for polyScore in polygonsWithScore:
                        poly = np.array([polyScore[0]], np.int32)
                        
                        if (flag==0):
                            polyID=polyScore[1]
                            flag=2
                        riskpoly=polyScore[2]
                       

                        inside = cv2.pointPolygonTest(poly, center, False)
                        # print(inside)
                        if inside == 1:
                            # print("Yes")
                            if (polyID!=prevpolyID):
                                count+=1
                                # print(count)
                                flag=0
                                if(riskpoly=='1'):
                                    riskcount+=1
                                    # print("risk =", riskcount)

                            prevpolyID=polyScore[1]
                            inside=5
                            cv2.drawContours(imgBoard, poly, -1, color=(0, 255, 0))
                            imgBoard = cv2.addWeighted(imgBoard, 0.7, imgBoard, 0.2, 0)

                imgBoard = cv2.addWeighted(imgBoard, 0.7, imgBoard, 0.5, 0)
                flag=0
                
                cv2.rectangle(imgBoard, (10, 2), (100,20), (255,255,255), -1)
                cv2.putText(imgBoard, str(cap.get(cv2.CAP_PROP_POS_FRAMES)), (15, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
                # cv2.imwrite('board.png',imgBoard)
                cv2.imshow('Frame',frame)
                cv2.imshow('Frame2',imgBoard)
                # cv2.imshow('Frame', frame)
                cv2.imshow("mask HSV",mask)
                out.write(imgBoard)
                if cv2.waitKey(10) & 0xFF == 27:
                    cv2.destroyAllWindows()
                    break
    print("Risk count ==> ", riskcount )
    openFieldResults.append(count)
    openFieldResults.append(riskcount)


def csv_downloader(data):
   csvfile=data.to_csv()
   b64=base64.b64encode(csvfile.encode()).decode()
   new_filename="new_text_file_{}_.csv".format(timestr)
   st.markdown("#### Download File ###")
   href=f'<a href="data: file/csv;base64, {b64}" download="{new_filename}"> Click Here !! </a>'
   st.markdown(href,unsafe_allow_html=True)

def showMainPageStudent():
    with mainSection:
        st.sidebar.title("Select Experment")
        choice  = st.sidebar.selectbox("Experment",("Morris water Maze","Open Field"))
        #["Show Instruction","Landmark identification","Show the #source code", "About"]

        
        if choice == "Morris water Maze":
            uploaded_video = st.file_uploader("Upload Video", type = ['mp4','mpeg','mov'])
            if uploaded_video != None:
                
                vid = uploaded_video.name
                with open(vid, mode='wb') as f:
                    f.write(uploaded_video.read()) # save video to disk

                st_video = open(vid,'rb')
                video_bytes = st_video.read()
                st.video(video_bytes)
                st.write("The Uploaded Video")
                start_button=st.button("Start")
                codeOnvideo=st.sidebar.checkbox("edit write results on video")
                if start_button:
                    # if codeOnvideo 
                    MorrisWaterMaze(vid)
                    DashboardMWM(getUserType())

                    st.text("hello water M")
                    try:
                        clip = moviepy.VideoFileClip('detected_video.mp4')
                        clip.write_videofile("myvideo.mp4")
                        st_video = open('myvideo.mp4','rb')
                        video_bytes = st_video.read()
                        st.video(video_bytes)
                        st.write("Detected Video") 
                    except OSError:
                        ''
 
def DashboardMWM(userType):
    global resultList
    if userType=="Doctor":
        dataresult=resultList[0]
        kpi1, kpi2, kpi3 = st.columns(3)

        with kpi1:
            st.markdown("**Number of entires**")
            kpi1_text = st.markdown("0")

        with kpi2:
            st.markdown("**Time inside target quadrant**")
            kpi2_text = st.markdown("0")

        with kpi3:
            st.markdown("**Time till to reach target**")
            kpi3_text = st.markdown("0")

        st.markdown("<hr/>", unsafe_allow_html=True)
        resultToString=[]
        for value in dataresult:
            # resultToString=value+": "+str(dataresult[value])
            resultToString.append(str(dataresult[value]))
            # st.header(resultToString)
        kpi1_text.write(f"<h1 style='text-align: center; color: red;'>{resultToString[0]}</h1>", unsafe_allow_html=True)
        kpi2_text.write(f"<h1 style='text-align: center; color: red;'>{resultToString[1]}</h1>", unsafe_allow_html=True)
        kpi3_text.write(f"<h1 style='text-align: center; color: red;'>{resultToString[2]}</h1>", unsafe_allow_html=True)
    elif userType=="Student":
        kpi1, kpi2, kpi3 = st.columns(3)

        with kpi1:
            st.markdown("**Number of entires**")
            st.text_input("enter Number of entires")

        with kpi2:
            st.markdown("**Time inside target quadrant**")
            st.text_input("enter Time inside target quadrant")

        with kpi3:
            st.markdown("**Time till to reach target**")
            st.text_input("enter Time till to reach target")

        st.markdown("<hr/>", unsafe_allow_html=True)
        Submit_button=st.button("Submit")
        if Submit_button:
            st.header("Thank you you data is saved")

def DashboardOpenFelid(userType):
    global resultList
    if userType=="Doctor":
        kpi1, kpi2, kpi3 = st.columns(3)

        with kpi1:
            st.markdown("**Crossings**")
            kpi1_text = st.markdown("0")

        with kpi2:
            st.markdown("**Risk**")
            kpi2_text = st.markdown("0")


        st.markdown("<hr/>", unsafe_allow_html=True)
        kpi1_text.write(f"<h1 style='text-align: center; color: red;'>{openFieldResults[0]}</h1>", unsafe_allow_html=True)
        kpi2_text.write(f"<h1 style='text-align: center; color: red;'>{openFieldResults[1]}</h1>", unsafe_allow_html=True)
    elif userType=="Student":
        kpi1, kpi2, kpi3 = st.columns(3)

        with kpi1:
            st.markdown("**Number of entires**")
            st.text_input("enter Number of entires")

        with kpi2:
            st.markdown("**Time inside target quadrant**")
            st.text_input("enter Time inside target quadrant")

        with kpi3:
            st.markdown("**Time till to reach target**")
            st.text_input("enter Time till to reach target")

        st.markdown("<hr/>", unsafe_allow_html=True)
        Submit_button=st.button("Submit")
        if Submit_button:
            st.header("Thank you you data is saved")

def showMainPageDoctor():
    with mainSection:
        st.sidebar.title("Select Experment")
        choice  = st.sidebar.selectbox("Experment",("Morris water Maze","Open Field"))
        #["Show Instruction","Landmark identification","Show the #source code", "About"]

        
        if choice == "Morris water Maze":
            uploaded_video = st.file_uploader("Upload Video", type = ['mp4','mpeg','mov'])
            if uploaded_video != None:
                
                vid = uploaded_video.name
                with open(vid, mode='wb') as f:
                    f.write(uploaded_video.read()) # save video to disk

                st_video = open(vid,'rb')
                video_bytes = st_video.read()
                st.video(video_bytes)
                st.write("The Uploaded Video")
                start_button=st.button("Start")
                codeOnvideo=st.checkbox("edit write results on video")
                if start_button:
                    # if codeOnvideo 

                    MorrisWaterMaze(vid)
                    DashboardMWM(getUserType())
                    

                    # st.text("hello water M")
                    try:
                        clip = moviepy.VideoFileClip('detected_video.mp4')
                        clip.write_videofile("myvideo.mp4")
                        st_video = open('myvideo.mp4','rb')
                        video_bytes = st_video.read()
                        st.video(video_bytes)
                        st.write("Detected Video") 
                    except OSError:
                        ''
        if choice == "Open Field":
            uploaded_video = st.file_uploader("Upload Video", type = ['mp4','mpeg','mov'])
            if uploaded_video != None:
                
                vid = uploaded_video.name
                with open(vid, mode='wb') as f:
                    f.write(uploaded_video.read()) # save video to disk

                st_video = open(vid,'rb')
                video_bytes = st_video.read()
                st.video(video_bytes)
                st.write("The Uploaded Video")
                start_button=st.button("Start")
                codeOnvideo=st.checkbox("edit write results on video")
                if start_button:
                    OpenField(vid)
                    DashboardOpenFelid(getUserType())
                    # st.text("hello water M")
                    try:
                        clip = moviepy.VideoFileClip('detected_video.mp4')
                        clip.write_videofile("myvideo.mp4")
                        st_video = open('myvideo.mp4','rb')
                        video_bytes = st_video.read()
                        st.video(video_bytes)
                        st.write("Detected Video") 
                    except OSError:
                        ''


def LoggedOut_Clicked():
    st.session_state['loggedIn'] = False
    
def show_logout_page():
    loginSection.empty()
    with logOutSection:
        st.sidebar.button ("Log Out", key="logout", on_click=LoggedOut_Clicked)

  

def LoggedIn_Clicked(userName, password):
    if login(userName, password):
        st.session_state['loggedIn'] = True
    else:
        st.session_state['loggedIn'] = False
        st.error("Invalid user name or password")

def login(email, password):
    global SelectedUser
    try:
        user=auth.sign_in_with_email_and_password(email,password)
        
        SelectedUser=user['localId']
        st.session_state['User'] = user['localId']

        return True
    except:
        return False 



def Signup_Clicked(username,email,password,UserType):
    global SelectedUser
    try:
      user=auth.create_user_with_email_and_password(email,password)
      st.success("Thank you for creating new account. Login PLease")
      user = auth.sign_in_with_email_and_password(email, password)
      data={"UserName":username,"Email":email,"Password":user['localId'],"UserType": UserType}
      db.child(user['localId']).set(data)
      SelectedUser=user['localId']
      st.session_state['loggedIn'] = True
      st.session_state['User'] = user['localId']

    except:
        st.error("field all data please")
def show_loginAndSignUp_page():

    with loginSection:
        if st.session_state['loggedIn'] == False:
            loginOrSignUp=st.selectbox('Login/Sign Up',['Login','Sign Up'])
            if loginOrSignUp=="Login":
                email=st.text_input("Email",placeholder="please enter your Email address")
                password=st.text_input("Password",placeholder="please enter your Password",type='password')
                st.button ("Login", on_click=LoggedIn_Clicked, args= (email, password))
            elif loginOrSignUp=="Sign Up":
                userName=st.text_input("UserName",placeholder="please enter your UserName")
                email=st.text_input("Email",placeholder="please enter your Email address")
                password=st.text_input("Password",placeholder="please enter your Password",type='password')
                UserType=st.selectbox('User Type',['Doctor','Student'])

                st.button ("Signup", on_click=Signup_Clicked, args= (userName,email, password,UserType))
    return email,password

def showWelcomeMessage():
    data=db.child(st.session_state['User']).get()
    print(data.val()["UserName"])
    st.sidebar.header("Welcome "+data.val()["UserName"])
    st.sidebar.subheader(data.val()["UserType"])

def getUserType():
    data=db.child(st.session_state['User']).get()
    return data.val()["UserType"]



with headerSection:
    st.title("iRats")
    #first run will have nothing in session_state
    if 'loggedIn' not in st.session_state:
        st.session_state['loggedIn'] = False
        st.session_state['Signup'] = False
        st.session_state['User'] = ""
        show_loginAndSignUp_page() 
        
    else:
        if st.session_state['loggedIn']:
            showWelcomeMessage()
            userType= getUserType()
            if userType=="Student":
                showMainPageStudent()
            elif userType=="Doctor":
                showMainPageDoctor()

            show_logout_page()  
           
        else:
            show_loginAndSignUp_page()

    # new_title = '<p style="font-size: 42px;">Welcome to my Object Detection App!</p>'
    # # read_me_0 = st.markdown(new_title, unsafe_allow_html=True)
        # choice  = st.sidebar.selectbox("Experment",("Morris water Maze","Open Field","Setting","About"))
