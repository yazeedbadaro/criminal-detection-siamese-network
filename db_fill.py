import random 
import pinecone
import os
from glob import glob
from utility.model_utils import *

pinecone.init(api_key= "5030e476-e093-4104-83d0-ec6f09ca7542", environment="northamerica-northeast1-gcp")
index = pinecone.Index("grad-index")
    
gender=["male","female"]
felony=["Robbery","Kidnapping","Arson","Burglary","Driving under the influence","Bribery","Assault","Human trafficking","Bribery","Assault"]
c=0
for folder in glob("archive (2)/*"):
    if folder == ".DS_Store":
        continue
    if folder =="README":
        continue
    
    for path in glob(f"{folder}/*.pgm"):
        index.upsert(
            vectors=[
                (
                f"{c}",               
                get_image_embedding(path).tolist(),
                {"label": folder.split("/")[-1],"path":path,"gender":random.choice(gender),"age":random.randint(20,60),"felony":random.choice(felony)}   
                )
            ]
        )
        print(f"Vector {c} upserted successfully!")
        c=c+1
