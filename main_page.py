import numpy as np
from PIL import Image
from utility.face_utils import *
from streamlit_option_menu import option_menu
import pinecone
import base64

st.set_page_config(
    page_title="criminal detection", 
    page_icon="👨‍🎓", 
)

with open("back.png", "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read())
st.markdown(
f"""
<style>
.stApp {{
    background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
    background-size: cover
}}
</style>
""",
unsafe_allow_html=True
)
    

pinecone.init(api_key= "5030e476-e093-4104-83d0-ec6f09ca7542", environment="northamerica-northeast1-gcp")
index = pinecone.Index("grad-index")

st.session_state.more_stuff = False

#main menu
selected = option_menu(None, ["How to use","Upload Image","Upload Video","Webcam"], 
    icons=['info-circle', 'card-image', "camera-video","webcam"], 
    menu_icon="cast", default_index=0, orientation="horizontal")

st.header(selected)

#How to use
if selected=="How to use":
    
    st.subheader("Upload an Image or Video")
    st.markdown("Simply :red[drag] and :blue[drop] your desired image or video")
    file_ = open("upload_gif_dark.gif", "rb")
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()

    st.markdown(
        f'<img src="data:image/gif;base64,{data_url}" alt="upload gif">',
        unsafe_allow_html=True,
    )
    st.markdown("")
    st.markdown("Once the file is done being processed a list of detected faces will appear.")
    st.subheader("Webcam")
    st.markdown("Click on the :red[RED START BUTTON] to start your camera then point it at the area where the detection will happen. Once you are done click on the 'detect faces' button and a list of detected faces will appear.")

#Image
if selected=="Upload Image":
    empty_files()
    uploaded_file = st.file_uploader("Choose an image",accept_multiple_files=False,type=["png","jpg","jpeg","pgm"])

    if uploaded_file is not None:
        st.image(uploaded_file)
        st.image(image_face_detector(np.array(Image.open(uploaded_file).convert("RGB")),n=1,conf_thresh=0))
        
        #display results
        for i,person in enumerate(glob("detected_faces/*.jpg")):
            query_response = index.query(
                                top_k=3,
                                include_values=False,
                                include_metadata=True,
                                vector=get_image_embedding(person).tolist(),
                            )
            
            with st.expander(f"Person {i+1}"):
                col1,col2= st.columns(2)
                with col1:
                    img=Image.open(person).resize((224,224))
                    st.subheader("Detected person image")
                    st.image(img) 
          
                with col2:
                    st.subheader("Criminal profile")
                    cols= st.columns(3)
                    
                    for cn,col in enumerate(cols):
                        
                        with col:
                            st.image(Image.open(query_response['matches'][cn]["metadata"]["path"]).resize((224,224)))
                            st.divider()
                            st.markdown("Name: "+query_response['matches'][cn]["metadata"]["label"])
                            st.markdown("Age: "+str(query_response['matches'][cn]["metadata"]["age"]))
                            st.markdown("Gender: "+str(query_response['matches'][cn]["metadata"]["gender"]))
                            st.markdown("Felony: "+str(query_response['matches'][cn]["metadata"]["felony"]))


#Video
if selected=="Upload Video":
    empty_files()
    uploaded_file = st.file_uploader("Choose a video",accept_multiple_files=False,type=["mp4"])

    if uploaded_file is not None:
        video_face_detector(uploaded_file)
        best_images("detected_faces",get_conf())
        
        #display results
        for i,person in enumerate(glob("detected_faces/highest_score_images/*.jpg")):
            query_response = index.query(
                                top_k=3,
                                include_values=False,
                                include_metadata=True,
                                vector=get_image_embedding(person).tolist(),
                            )
            
            with st.expander(f"Person {i+1}"):
                col1,col2= st.columns(2)
                with col1:
                    img=Image.open(person).resize((224,224))
                    st.subheader("Detected person image")
                    st.image(img) 
          
                with col2:
                    st.subheader("Criminal profile")
                    cols= st.columns(3)
                    
                    for cn,col in enumerate(cols):
                        
                        with col:
                            st.image(Image.open(query_response['matches'][cn]["metadata"]["path"]).resize((224,224)))
                            st.divider()
                            st.markdown("Name: "+query_response['matches'][cn]["metadata"]["label"])
                            st.markdown("Age: "+str(query_response['matches'][cn]["metadata"]["age"]))
                            st.markdown("Gender: "+str(query_response['matches'][cn]["metadata"]["gender"]))
                            st.markdown("Felony: "+str(query_response['matches'][cn]["metadata"]["felony"]))

    
#webcam
if selected=="Webcam":

    webrtc_streamer(key="example",video_frame_callback=video_frame_callback,media_stream_constraints={"video": True, "audio": False})
    
    click = st.button("Detect Faces")
    if click:
        st.session_state.more_stuff = True

    if st.session_state.more_stuff:
        best_images("detected_faces",get_conf())
        #display results
        for i,person in enumerate(glob("detected_faces/highest_score_images/*.jpg")):
            query_response = index.query(
                                top_k=3,
                                include_values=False,
                                include_metadata=True,
                                vector=get_image_embedding(person).tolist(),
                            )
            
            with st.expander(f"Person {i+1}"):
                col1,col2= st.columns(2)
                with col1:
                    img=Image.open(person).resize((224,224))
                    st.subheader("Detected person image")
                    st.image(img) 
          
                with col2:
                    st.subheader("Criminal profile")
                    cols= st.columns(3)
                    
                    for cn,col in enumerate(cols):
                        
                        with col:
                            st.image(Image.open(query_response['matches'][cn]["metadata"]["path"]).resize((224,224)))
                            st.divider()
                            st.markdown("Name: "+query_response['matches'][cn]["metadata"]["label"])
                            st.markdown("Age: "+str(query_response['matches'][cn]["metadata"]["age"]))
                            st.markdown("Gender: "+str(query_response['matches'][cn]["metadata"]["gender"]))
                            st.markdown("Felony: "+str(query_response['matches'][cn]["metadata"]["felony"]))
    empty_files()
