# USAGE
# python test_network.py --model=m1.model --label=m_label.txt --image=valData/z_test_10.png

# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2

# construct the argument parse and parse the arguments
#ap = argparse.ArgumentParser()
#ap.add_argument("-m", "--model", required=True,
#	help="path to trained model model")
#ap.add_argument("-l", "--label", required=True,
#	help="path to label file")
#ap.add_argument("-i", "--image", required=True,
#	help="path to input image")
#args = vars(ap.parse_args())

arg_model = "bw.model"
arg_label = "bw_label.txt"
arg_image = "testingData/J/j_test_4.jpg"

labels = []

# load the image
image = cv2.imread(arg_image)
orig = image.copy()

# pre-process the image for classification
image = cv2.resize(image, (80, 96))
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

# load the trained convolutional neural network
print("[INFO] loading network...")
model = load_model(arg_model)

# classify the input image
#(J, Z) = model.predict(image)[0]
prediction = model.predict(image)[0]

## read labels from the training
with open(arg_label, 'r') as f:
    for line in f:
        labels.append(line.strip())

index_max = np.argmax(prediction)
label = "{}: {:.2f}%".format(labels[index_max], prediction[index_max] * 100)
print("##########")
print("  RESULT")
print("##########")
print(label)

# build the label
#print(J, Z)
#print ("#####")
#label = "J" if J > Z else "Z"
#print(label)
#proba = J if J > Z else Z
#label = "{}: {:.2f}%".format(label, proba * 100)

# draw the label on the image
output = imutils.resize(orig, width=400)
cv2.putText(output, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,
	0.7, (0, 255, 0), 2)

# show the output image
cv2.imshow("Output", output)
cv2.waitKey(0)