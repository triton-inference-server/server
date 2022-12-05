import numpy as np
import tritonclient.http as httpclient
from tritonclient.utils import triton_to_np_dtype

import math
import cv2

def fourPointsTransform(frame, vertices):
    vertices = np.asarray(vertices)
    outputSize = (100, 32)
    targetVertices = np.array([
        [0, outputSize[1] - 1],
        [0, 0],
        [outputSize[0] - 1, 0],
        [outputSize[0] - 1, outputSize[1] - 1]], dtype="float32")

    rotationMatrix = cv2.getPerspectiveTransform(vertices, targetVertices)
    result = cv2.warpPerspective(frame, rotationMatrix, outputSize)
    return result

def decodeBoundingBoxes(scores, geometry, scoreThresh):
    detections = []
    confidences = []

    ############ CHECK DIMENSIONS AND SHAPES OF geometry AND scores ############
    assert len(scores.shape) == 4, "Incorrect dimensions of scores"
    assert len(geometry.shape) == 4, "Incorrect dimensions of geometry"
    assert scores.shape[0] == 1, "Invalid dimensions of scores"
    assert geometry.shape[0] == 1, "Invalid dimensions of geometry"
    assert scores.shape[1] == 1, "Invalid dimensions of scores"
    assert geometry.shape[1] == 5, "Invalid dimensions of geometry"
    assert scores.shape[2] == geometry.shape[2], "Invalid dimensions of scores and geometry"
    assert scores.shape[3] == geometry.shape[3], "Invalid dimensions of scores and geometry"
    height = scores.shape[2]
    width = scores.shape[3]
    for y in range(0, height):
    # Extract data from scores
        scoresData = scores[0][0][y]
        x0_data = geometry[0][0][y]
        x1_data = geometry[0][1][y]
        x2_data = geometry[0][2][y]
        x3_data = geometry[0][3][y]
        anglesData = geometry[0][4][y]
        for x in range(0, width):
            score = scoresData[x]
            # If score is lower than threshold score, move to next x
            if (score < scoreThresh):
                continue
            # Calculate offset
            offsetX = x * 4.0
            offsetY = y * 4.0
            angle = anglesData[x]

            # Calculate cos and sin of angle
            cosA = math.cos(angle)
            sinA = math.sin(angle)
            h = x0_data[x] + x2_data[x]
            w = x1_data[x] + x3_data[x]

            # Calculate offset
            offset = ([offsetX + cosA * x1_data[x] + sinA * x2_data[x], offsetY - sinA * x1_data[x] + cosA * x2_data[x]])

            # Find points for rectangle
            p1 = (-sinA * h + offset[0], -cosA * h + offset[1])
            p3 = (-cosA * w + offset[0], sinA * w + offset[1])
            center = (0.5 * (p1[0] + p3[0]), 0.5 * (p1[1] + p3[1]))
            detections.append((center, (w, h), -1 * angle * 180.0 / math.pi))
            confidences.append(float(score))

    # Return detections and confidences
    return [detections, confidences]

# read image
img = cv2.imread("./img1.jpg")
height_ = img.shape[0]
width_ = img.shape[1]
confThreshold = 0.5
nmsThreshold = 0.4
inpWidth = 640
inpHeight = 480

rW = width_ / float(inpWidth)
rH = height_ / float(inpHeight)

# pre-process image
blob = cv2.dnn.blobFromImage(img, 1.0, (inpWidth, inpHeight), (123.68, 116.78, 103.94), True, False)
blob = np.transpose(blob, (0, 2,3,1))

# Setting up client
client = httpclient.InferenceServerClient(url="localhost:8000")

input_image = httpclient.InferInput("input_images:0", blob.shape, datatype="FP32")
input_image.set_data_from_numpy(blob, binary_data=True)

scores = httpclient.InferRequestedOutput("feature_fusion/Conv_7/Sigmoid:0", binary_data=True)
geometry = httpclient.InferRequestedOutput("feature_fusion/concat_3:0", binary_data=True)

# Querying the server
query = client.infer(model_name="text_detection", inputs=[input_image], outputs=[scores, geometry])

scores_ = np.transpose(query.as_numpy('feature_fusion/Conv_7/Sigmoid:0'), (0,3,1,2))
geometry_ = np.transpose(query.as_numpy('feature_fusion/concat_3:0'), (0,3,1,2))
print(scores_.shape)

#print(scores_.shape)
[boxes, confidences] = decodeBoundingBoxes(scores_, geometry_, confThreshold)
indices = cv2.dnn.NMSBoxesRotated(boxes, confidences, confThreshold, nmsThreshold)
ctr = 0
for i in indices:
    vertices = cv2.boxPoints(boxes[i])
    for j in range(4):
        vertices[j][0] *= rW
        vertices[j][1] *= rH

    cropped = fourPointsTransform(img, vertices)
    cv2.imwrite(str(ctr)+".jpg",cropped)
    ctr+=1
