import streamlit as st
import pickle
from pathlib import Path
import streamlit_authenticator as stauth
from PIL import Image
import base64
from io import BytesIO
import os
from yoloface import face_analysis
import numpy
import cv2
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import pinecone
from tensorflow.keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt



def yolo_model_image(video_dir):   
    for video_file in os.listdir(video_dir):
        if video_file.endswith('.jpg'):
            print('Processing video:', video_file)
            cap = cv2.VideoCapture(os.path.join(video_dir, video_file))
            i = 0
            while True: 
                ret, frame = cap.read()
                if not ret:
                    break
                _, box, conf = face.face_detection(frame_arr=frame, frame_status=True, model='tiny')
                for (x, y, w, h) in box:
                    roi_color = frame[y:y+h, x:x+w]
                    print("[INFO] Object found. Saving locally.")
                    try:
                        cv2.imwrite(os.path.join(output_dir, f'{video_file}_frame{i}_face.jpg'), roi_color)
                    except:
                        pass
                    i += 1
                # display output frames
                # output_frame = face.show_output(img=frame, face_box=box, frame_status=True)
                # cv2.imshow('frame', output_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            cap.release()
            cv2.destroyAllWindows()

#################################################################################

def yolo_model_video(video_dir):   
    # Define the path to the folder for highest score images
    high_score_folder = os.path.join(faces_dir, 'highest_score_images')
    if not os.path.exists(high_score_folder):
        os.mkdir(high_score_folder)

    for video_file in os.listdir(video_dir):
        if not video_file.endswith('.mp4'):
            continue
            
        print('Processing video:', video_file)
        
        # Reset variables
        confidences = []
        images = []
        orig_images = []
        
        video_folder = os.path.join(faces_dir, f'{os.path.splitext(video_file)[0]}')
        if not os.path.exists(video_folder):
            os.mkdir(video_folder)
        cap = cv2.VideoCapture(os.path.join(video_dir, video_file))
        i = 0
        while True: 
            ret, frame = cap.read()
            if not ret:
                break
            _, box, conf = face.face_detection(frame_arr=frame, frame_status=True, model='tiny')
            for (x, y, w, h) in box:
                roi_color = frame[y:y+h, x:x+w]
                print("[INFO] Object found. Saving locally.")
                try:
                    cv2.imwrite(os.path.join(video_folder, f'{os.path.splitext(video_file)[0]}_frame{i}_face.jpg'), roi_color)
                    confidences.append(conf)  # Append the confidence value to the list
                except:
                    pass
                i += 1
            # display output frames
            output_frame = face.show_output(img=frame, face_box=box, frame_status=True)
            #cv2.imshow('frame', output_frame)
            plt.imshow(output_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

        # Load all images from the 'faces' folder and preprocess them for ResNet50
        for filename in os.listdir(video_folder):
            img = cv2.imread(os.path.join(video_folder, filename))
            orig_images.append(img)
            try:
                img = cv2.resize(img, (224, 224))
            except:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = preprocess_input(img)
            images.append(img)

        # Convert the list of images to a 4D numpy array
        X = np.array(images)

        # Apply ResNet50 feature extraction to the image data
        model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
        features = model.predict(X)

        # Flatten the features into a 2D numpy array
        features = features.reshape(features.shape[0], -1)

        # Calculate cosine similarity matrix
        cos_sim = cosine_similarity(features)

        # Apply hierarchical clustering to the feature data
        hierarchical = AgglomerativeClustering(n_clusters=None, linkage='average', distance_threshold=0.8)
        hierarchical.fit(cos_sim)

        # Get the labels assigned to each image
        labels = hierarchical.labels_.astype(int)

        # Process images from each cluster
        for i in range(np.max(labels) + 1):
            print(f'Processing images in cluster {i}...')
            # Get the indices of the images in this cluster
            indices = np.where(labels == i)[0]
            # Calculate the score for each image in this cluster
            scores = []
            for idx in indices:
                img = orig_images[idx]
                brightness_score = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean()
                contrast_score = np.std(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
                sharpness_score = cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
                confidence = confidences[idx]  # get the confidence value for this image
                score = (0.5 * brightness_score + 0.3 * contrast_score + 0.2 * sharpness_score) * confidence[0]
                scores.append(score)
            # Save the image with the highest score to the appropriate folder
            highest_score_idx = np.argmax(scores)
            highest_score_img = orig_images[indices[highest_score_idx]]
            filename = os.path.join(high_score_folder, f'score_{scores[highest_score_idx]:.4f}_{indices[highest_score_idx]}.jpg')
            cv2.imwrite(filename, highest_score_img)



#########################################################################################
new_model = tf.keras.models.load_model('siamese_resnet.h5',compile=False)
pinecone.init(api_key= "5030e476-e093-4104-83d0-ec6f09ca7542", environment="northamerica-northeast1-gcp")
index = pinecone.Index("grad")

def get_image_embedding(image_path):
    img = Image.open(image_path).resize((224, 224))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    embedding = new_model.predict(x)
    return embedding.flatten().tolist()


def res(imgP):
    results = index.query(
        top_k=10, 
        include_values=False, 
        include_metadata=True, 
        vector=get_image_embedding(imgP))
    matches = results['matches']
    paths = [match['metadata']['path'] for match in matches]

    return paths

#########################################################################################

            
# Find more emojis here: https://www.webfx.com/tools/emoji-cheat-sheet/
st.set_page_config(
    page_title="criminal detection", 
    page_icon="👨‍🎓", 
    layout="wide" 
)
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
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
add_bg_from_local('back.png')    

# ---- user authentication ----
names = ['yousef','razan','yazeed']
usernames = ['yhamdan','ralshabah','ybdaro']

file_path = Path(__file__).parent / "hashed_pw.pkl"
with file_path.open('rb') as file:
    hashed_passwords = pickle.load(file)

authenticator = stauth.Authenticate(names,usernames,hashed_passwords,
                                    "user_dashboard","abcdef",cookie_expiry_days=30)

name,authentication_status,username = authenticator.login("login","main")

if authentication_status == False:
    st.error("username/password is incorrect")

if authentication_status == None:
    st.warning("please enter your username and password")

if authentication_status:
    # ---- HEADER SECTION ----
    #adding a button
    

    button_choice = st.sidebar.radio("Choose an option:", ["Upload image", "Upload video", "How to use"])

    if button_choice == "Upload image":
        i=0
        # Use st.file_uploader to create a file uploader widget
        uploaded_files = st.file_uploader("Choose image files", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

        # Check if the user has uploaded files
        if uploaded_files is not None:
            for uploaded_file in uploaded_files:
                # Use PIL to open the uploaded image and display it
                image = Image.open(uploaded_file)
                st.image(image, caption='Uploaded Image')
                # Use PIL to save the image to a file
                image = Image.open(uploaded_file)
                image.save(f'imgORvid/saved_image_{i}.png')
                i=i+1
                if st.button("detect faces"):
                    image_dir = 'imgORvid'
                    output_dir = 'faces/fromimage'
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    yolo_model_image(image_dir)
                if st.button("results"):
                    output_dir = 'faces/fromimage'
                    # Loop through all files in the directory and display each image
                    for filename in os.listdir(output_dir):
                        if filename.endswith(".jpg") or filename.endswith(".png"):
                            filepath = os.path.join(output_dir, filename)
                            image = Image.open(filepath)
                            st.image(image, caption=filename)
            

            
    elif button_choice == "Upload video":
        i=0
        # Use st.file_uploader to create a file uploader widget for videos
        uploaded_files = st.file_uploader("Choose video files", type=["mp4", "avi", "mkv"], accept_multiple_files=True)

        # Create an empty list to store the uploaded videos
        video_list = []

        # Check if the user has uploaded any files
        if uploaded_files is not None:
            for uploaded_file in uploaded_files:
                # Append each uploaded video to the list
                video_list.append(uploaded_file)

        # Display all uploaded videos on the screen
        for uploaded_video in video_list:
            st.video(uploaded_video)
            with open(f'imgORvid/saved_video_{i}.mp4', 'wb') as f:
                f.write(uploaded_video.read())
            i += 1
        if st.button("detect faces"):
            face = face_analysis()
            video_dir = 'imgORvid'
            faces_dir = 'faces'
            if not os.path.exists(faces_dir):
                os.makedirs(faces_dir)
            yolo_model_video(video_dir)


            
        if st.button("results"):
            output_dir = 'faces/highest_score_images'
            # Loop through all files in the directory and display each image
            for filename in os.listdir(output_dir):
                folder_paths = {}
                if filename.endswith(".jpg") or filename.endswith(".png"):
                    filepath = os.path.join(output_dir, filename)
                    image = Image.open(filepath)
                    st.image(image, caption=filename)
                    paths = res(filepath)
                    for path in paths:
                        folder_name = os.path.dirname(path)
                        if folder_name not in folder_paths:
                            folder_paths[folder_name] = path
                            image = Image.open(path)
                            st.image(image, caption=path)



        if st.button("finish"):
            output_dir = 'faces/highest_score_images'
            image_dir = 'imgORvid'
            # Use a for loop to iterate through the contents of the folder
            for item in os.listdir(output_dir):
                # Use os.path.join() to create the full path to each item in the folder
                item_path = os.path.join(output_dir, item)
                # Use os.remove() to delete each file in the folder
                if os.path.isfile(item_path):
                    os.remove(item_path)

            for video_file in os.listdir(image_dir):
                if video_file.endswith('.mp4'):
                    v_path = os.path.join(image_dir, video_file)
                    # Use os.remove() to delete each file in the folder
                    if os.path.isfile(v_path):
                        os.remove(v_path)
                            


                    



    elif button_choice == "How to use":
        st.write("Instructions on how to use the app.")




    # st.subheader("Hi, I am Sven :wave:")
    # st.title("A Data Analyst From Germany")
    # st.write("I am passionate about finding ways to use Python and VBA to be more efficient and effective in business settings.")
    # st.write("[Learn More >](https://pythonandvba.com)")
    st.sidebar.title(f"welcom {name}")
    authenticator.logout('logout','sidebar')


