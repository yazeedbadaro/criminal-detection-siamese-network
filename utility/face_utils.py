import streamlit as st
import cv2
import os
import subprocess
from streamlit_webrtc import webrtc_streamer
import av
from ultralytics import YOLO

face=YOLO("yolov8n-face.pt")

def image_face_detector(image,n,flag=True):
    frame=image.copy()
    results=face(frame)
    boxes=results[0].boxes.xyxy
    i=1
    
    for box in boxes:
        (x1,y1,x2,y2)=[int(x) for x in box.tolist()]
        cv2.rectangle(frame,(x1,y1),(x2,y2),(255, 0, 0),2)
        (w, h), _ = cv2.getTextSize("face", cv2.FONT_HERSHEY_DUPLEX, 0.6, 1)

        frame = cv2.rectangle(frame, (x1, y1 - 20), (x1+ w, y1), (255, 0, 0), -1)
        frame = cv2.putText(frame, "face", (x1, y1 - 5),cv2.FONT_HERSHEY_DUPLEX, 0.6, (255,255,255), 1)

        try:
            if flag:
                cv2.imwrite(f"detected_faces/frame{n}_face{i}.jpg",image[y1:y2, x1:x2,::-1])
            else:
                cv2.imwrite(f"detected_faces/frame{n}_face{i}.jpg",image[y1:y2, x1:x2])
            i=i+1
        except:
            print("empty frame error")
    return frame

def video_face_detector(uploaded_file):
    path=os.path.join("uploaded_videos",uploaded_file.name)
    
    with open(path,"wb") as f:
         f.write(uploaded_file.getbuffer())
    cap = cv2.VideoCapture(path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter("video_detection/output.mp4",fourcc, int(cap.get(cv2.CAP_PROP_FPS)), (int(cap.get(3)),int(cap.get(4))))
    
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")
            
    my_bar = st.progress(0, text="Processing the video. Please Wait.")
    counter=0
    n_frame=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    while(cap.isOpened()):
        ret, frame = cap.read()
        
        if ret == True:
            out.write(image_face_detector(frame,counter+1,False))
            my_bar.progress((counter + 1)/n_frame, text="Processing the video. Please Wait.")
            counter=counter+1
        else:
            break
        
    cap.release()
    out.release()
    
    convertedVideo = "video_detection/testh264.mp4"
    subprocess.call(args=f"ffmpeg -y -i video_detection/output.mp4 -c:v libx264 {convertedVideo}".split(" "))
    st.video(convertedVideo)
    
def video_frame_callback(frame):
    if not hasattr(video_frame_callback, "frame_number"):
        video_frame_callback.frame_number = 1

    frame = frame.to_ndarray(format="bgr24")

    processed = image_face_detector(frame, video_frame_callback.frame_number, False)

    video_frame_callback.frame_number += 1

    return av.VideoFrame.from_ndarray(processed, format="bgr24")