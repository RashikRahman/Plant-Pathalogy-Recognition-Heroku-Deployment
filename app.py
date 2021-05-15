from pywebio.platform.flask import start_server, webio_view
from pywebio import STATIC_PATH
from flask import Flask, send_from_directory
from pywebio.input import *
from pywebio.output import *
import cv2
import time
from PIL import Image
import helper
import argparse


import numpy as np

app = Flask(__name__)


def predict():
    with popup("Plant Pathalogy Prediction"):
        put_text("Good to see you again, the prediction may take upto 7 sec depending on image resulation. Have patience.")

    img = file_upload("Select a image:", accept="images/*")

   

    put_processbar('bar')
    for i in range(1, 11):
        set_processbar('bar', i / 10)
        time.sleep(0.2)

    content = img['content']
    nparr = np.fromstring(content, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    pred = helper.main(img,model)    
    put_text('Predicted outcome: ',pred)    
    # put_markdown(pred)
    put_image(content)






app.add_url_rule('/tool', 'webio_view', webio_view(predict),
            methods=['GET', 'POST', 'OPTIONS'])
            
# app.run(host='localhost', port=80)


if __name__ == '__main__':
    global model 
    model = helper.load_model()
    parser = argparse.ArgumentParser()
    parser.add_argument("-p","--port",type=int,default=8080)
    args = parser.parse_args()

    start_server(predict,port=args.port)
    
