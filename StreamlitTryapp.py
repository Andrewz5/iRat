import tempfile
import tempfile
import cv2
import streamlit as st
from streamlit_webrtc import VideoProcessorBase

@st.cache()
def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized

def main():
    st.title("iRats")
    st.sidebar.title("Tools")
    st.sidebar.markdown("---")
    confidance=st.sidebar.slider('confidence',min_value=0.0,max_value=1.0,value=0.3)
    st.sidebar.markdown("---")
    st.set_option('deprecation.showfileUploaderEncoding', False)

    saveImg=st.sidebar.checkbox("Save Video")
    enableGPU=st.sidebar.checkbox("Enable GPU")
    use_webcam=st.sidebar.button('Live Streaming')
    record=st.sidebar.checkbox('Record video')
        
    stframe=st.empty()
    videoFile=st.sidebar.file_uploader("Upload a video", type=[ 'mp4', 'mov','avi','asf', 'm4v' ])
    tfflie = tempfile.NamedTemporaryFile(delete=False)
    if not videoFile:
        if use_webcam:
            video=cv2.VideoCapture(0)

    else:    
        tfflie.write(videoFile.read())
        video=cv2.VideoCapture(tfflie.name)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_input = int(video.get(cv2.CAP_PROP_FPS))
         #Recording
        #codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        codec = cv2.VideoWriter_fourcc('V','P','0','9')
        out = cv2.VideoWriter('output1.mp4', codec, fps_input, (width, height))
        st.sidebar.text('Input Video')
        st.sidebar.video(tfflie.name)
        fps = 0
        i = 0
        # drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=2)
        kpi1, kpi2, kpi3 = st.columns(3)

        with kpi1:
            st.markdown("**FrameRate**")
            kpi1_text = st.markdown("0")

        with kpi2:
            st.markdown("**Detected Faces**")
            kpi2_text = st.markdown("0")

        with kpi3:
            st.markdown("**Image Width**")
            kpi3_text = st.markdown("0")

        st.markdown("<hr/>", unsafe_allow_html=True)
        prevTime=0
        while video.isOpened():
            i+=1
            ret,frame=video.read()
            if not ret:
                continue
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame.flags.writeable = True
            frame = cv2.resize(frame,(0,0),fx = 0.8 , fy = 0.8)
            frame = image_resize(image = frame, width = 640)
            stframe.image(frame,channels = 'BGR',use_column_width=True)

        # codec=cv2.VideoWriter()
        # demoVideo=open(tfflie.name,'rb')
        # demoBytes=demoVideo.read()
        # st.sidebar.text("Input video")
        # st.sidebar.video(demoBytes)
        # # vid = cv2.VideoCapture(tfflie.name)
        # st.text(demoBytes)
        # stframe = st.empty()
        # cap = cv2.VideoCapture(tfflie.name)
        # while(1):
        #     ret,frame=cap.read()
        #     frame=cv2.COLOR_BAYER_BG2GRAY()
        #     stframe.image(frame,channels = 'BGR',use_column_width=True)

        # cap=cv2.VideoCapture(tfflie.name)
        # vid_obj = cv2.VideoCapture(tfflie.name) 
        # success = True

        # frame = st.empty()

        # while success:
        #     success, image = vid_obj.read()
        #     frame.image(image, channels="BGR")  # OpenCV images are BGR!
            
   
    
    # st.video(demoBytes)

    videoFile.release()


if __name__=="__main__":
    try:
        main()
    except:
        pass