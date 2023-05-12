import numpy as np
from PIL import Image
from utility.face_utils import *
from streamlit_option_menu import option_menu

#main menu
selected = option_menu(None, ["How to use","Upload Image","Upload Video","Webcam"], 
    icons=['info-circle', 'card-image', "camera-video","webcam"], 
    menu_icon="cast", default_index=0, orientation="horizontal")

st.header(selected)

#How to use
if selected=="How to use":
    
    st.subheader("Upload an Image")
    st.markdown("hello")
    st.subheader("Upload a Video")
    st.markdown("hello")

#Image
if selected=="Upload Image":
    uploaded_file = st.file_uploader("Choose an image",accept_multiple_files=False,type=["png","jpg","jpeg"])

    if uploaded_file is not None:
        st.image(uploaded_file)
        st.image(image_face_detector(np.array(Image.open(uploaded_file)),n=1))

#Video
if selected=="Upload Video":
    uploaded_file = st.file_uploader("Choose a video",accept_multiple_files=False,type=["mp4"])

    if uploaded_file is not None:
        video_face_detector(uploaded_file)
        
#webcam
if selected=="Webcam":
    webrtc_streamer(key="example",video_frame_callback=video_frame_callback,media_stream_constraints={"video": True, "audio": False})
