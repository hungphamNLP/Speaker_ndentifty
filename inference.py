import torch
import torch.nn.functional as F
import numpy as np
from model import net,device
from validation import test
from training import main
import os
from spectogram_extraction import extract_fbanks,preprocessing_adduser,processing_img
from fastapi import FastAPI, File, UploadFile, Request, Form, HTTPException
from fastapi.templating import Jinja2Templates 
from fastapi.staticfiles import StaticFiles
import uvicorn
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

DATA_DIR = 'data_files/'
MODEL_PATH = 'siamese_net_crossEntropy_withDropout.pt'

model_instance = net
model_instance.load_state_dict(torch.load(MODEL_PATH, map_location=device))
# model_instance = model_instance.double()
model_instance.eval()


def _save_file(request_, username):
    file = request_.files['file']
    dir_ = DATA_DIR + username
    if not os.path.exists(dir_):
        os.makedirs(dir_)

    filename = DATA_DIR + username + '/sample.wav'
    file.save(filename)
    return filename


@app.post('/add_speaker')
async def add_speaker(file:UploadFile=File(...),username: str= Form(...)):
    content = await file.read()
    audioPath = f"./static/record/file.wav"
    with open(audioPath, 'wb') as f: 
            f.write(content)
    preprocessing_adduser(audioPath,username)
    return 'sucess'

@app.get('/all_speaker')
async def show_all_speaker():
    list_user=os.listdir('data_dir')
    return {"all_user":(str(list_user))}

@app.post('/auth_speaker')    
async def inference_speaker_auth(audio:UploadFile=File(...),username_other:str= Form(...)):
    content = await audio.read()
    audioPath = f"./static/record/file.wav"
    with open(audioPath, 'wb') as f: 
            f.write(content)
    input1 = processing_img(username_other)
    input1 = input1.unsqueeze(0)
    user_search= 'user_test'
    preprocessing_adduser(audioPath,user_search)
    input2 = processing_img(user_search)
    input2 = input2.unsqueeze(0)
    rs=model_instance(input1,input2)
    print(rs)
    return {"distant speaker":str(rs[0][-1].detach().numpy())}
    
@app.post('/train_speaker')
async def train_speaker(epoch:int=Form(...)):
    main(epoch)
    return {"done"}


# @app.post('/test_speaker')
# async def test_speaker():
    

if __name__ == '__main__':
    # inference_speaker_auth('sample-0.wav','hungpham23')
    # # preprocessing('file_new.wav')
    uvicorn.run(app, port = 3000)