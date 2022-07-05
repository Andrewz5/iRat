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
#============================================================================== 
#== Streamlit Start
# =============================================================================
headerSection = st.container()
mainSection = st.container()
loginSection = st.container()
logOutSection = st.container()

def MorrisWaterMaze(video):
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
            print(frame.shape)
            frame=ROI(frame)
            print(frame.shape)
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
            # TimeSpentQuadrant= 
            # cv2.putText(frame, 'Number of entries %i'%(ratDetectionMethods.getCountNumberofEnter()),(50,400), cv2.FONT_HERSHEY_DUPLEX, 0.5,(0,0,255),2)
            # cv2.putText(frame, 'Time inside target quadrant %.2f sec'%((quarterFrame/60)),(50,420), cv2.FONT_HERSHEY_DUPLEX, 0.5,(0,0,255),2)
            # cv2.putText(frame, 'Time till reaching target quadrant  %.2f sec'%((latancyframe/60)),(50,440), cv2.FONT_HERSHEY_DUPLEX, 0.5,(0,0,255),2)
            #show image
            # cv2.imshow('frame',frame)
            # cv2.imshow('frame_c',frame_c)
            # cv2.imshow('mask',fgMask)
            out.write(frame)
    
        
        else:
            break
 
def showMainPageStudent():
    with mainSection:
        st.sidebar.title("Select Activity")
        choice  = st.sidebar.selectbox("Menu",("Morris water Maze","Setting","About"))
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
            showMainPageStudent()  
            show_logout_page()  
           
        else:
            show_loginAndSignUp_page()

    # new_title = '<p style="font-size: 42px;">Welcome to my Object Detection App!</p>'
    # # read_me_0 = st.markdown(new_title, unsafe_allow_html=True)
