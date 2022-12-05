import numpy as np
import tritonclient.http as httpclient
from tritonclient.utils import triton_to_np_dtype

import math
import cv2


def decodeText(scores):
    text = ""
    alphabet = "0123456789abcdefghijklmnopqrstuvwxyz"
    for i in range(scores.shape[0]):
        c = np.argmax(scores[i][0])
        if c != 0:
            text += alphabet[c - 1]
        else:
            text += '-'
    # adjacent same letters as well as background text must be removed to get the final output
    char_list = []
    for i in range(len(text)):
        if text[i] != '-' and (not (i > 0 and text[i] == text[i - 1])):
            char_list.append(text[i])
    return ''.join(char_list)

# read image
img = cv2.imread("./0.jpg")

# pre-process image
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blob = cv2.dnn.blobFromImage(gray, size=(100,32), mean=127.5, scalefactor=1 / 127.5)
print(blob.shape)

#blob = np.transpose(blob, (0, 2,3,1))

# Setting up client
client = httpclient.InferenceServerClient(url="localhost:8000")

input_image = httpclient.InferInput("input.1", blob.shape, datatype="FP32")
input_image.set_data_from_numpy(blob, binary_data=True)

text_matrix = httpclient.InferRequestedOutput("308", binary_data=True)

# Querying the server
query = client.infer(model_name="text_recognition", inputs=[input_image], outputs=[text_matrix])
text = decodeText(np.transpose(query.as_numpy('308'), (1,0,2)))
