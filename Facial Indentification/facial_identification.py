import cv2
import numpy as np
import os
from PIL import Image

subjects = ["", "DatGatto", "Celine Farach"]
faces = [] #list to hold all subject faces
labels = [] #list to hold labels for all subjects

def detect_face(img):
	
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #convert the test image to gray 	 
	#load OpenCV LBP face detector
	face_cascade = cv2.CascadeClassifier('lbpcascade_frontalface.xml')
	 
	#detect multiscale images, result is a list of faces
	faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);
	 
	#if no faces are detected then return original img
	if (len(faces) == 0):
		return None, None
	 
	#under the assumption that there will be only one face,
	#extract the face area
	(x, y, w, h) = faces[0]
	 
	#return only the face part of the image
	return gray[y:y+w, x:x+h], faces[0]

def prepare_training_data(data_folder_path):
	dirs = os.listdir(data_folder_path)
	for dir_name in dirs:
		if not dir_name.startswith("s"):
			#subject directories start with letter 's' so ignore any non-relevant directories if any
			continue	
		#extract label number of subject from dir_name, format of dir name = slabel, so removing letter 's' from dir_name will give us label
		label = int(dir_name.replace("s", ""))
		subject_dir_path = data_folder_path + "/" + dir_name #build path of directory containing images for current subject
		subject_images_names = os.listdir(subject_dir_path) #get the images names that are inside the given subject directory	
		for image_name in subject_images_names:	 
			if image_name.startswith("."): #ignore system files like .DS_Store
				continue;
			image_path = subject_dir_path + "/" + image_name #build image path
			image = cv2.imread(image_path) #read image
			#display an image window to show the image 
			cv2.imshow("Training on image...", image)
			cv2.waitKey(100)
			face, rect = detect_face(image)
			if face is not None:
				faces.append(face)
				labels.append(label)
			cv2.destroyAllWindows()
			cv2.waitKey(1)
			cv2.destroyAllWindows()
	return faces, labels
def train_face_indentifier():	
	face_recognizer.train(faces, np.array(labels))	#train our face recognizer of our training faces

def draw_rectangle(img, rect):
	#TODO:draw rectangle on image according to given (x, y) coordinates and given width and heigh in rect
	(x, y, w, h) = rect
	cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
 
def draw_text(img, text, x, y):
	#TODO: function to draw text on give image starting from passed (x, y) coordinates. 
	cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)


def predict(test_img):
	#TODO: recognizes the person in image then draws a rectangle around detected face with name of the subject
	img = test_img.copy() #make a copy of the image instead of change the original one
	face, rect = detect_face(img) #detect face from the image
 
	label= face_recognizer.predict(face) #predict the image using our face recognizer 

	label_text = subjects[label[0]] #get name of respective label returned by face recognizer
	draw_rectangle(img, rect)
	draw_text(img, label_text, rect[0], rect[1]-5)
 
	return img

print("Preparing data...")
faces, labels = prepare_training_data("training_data")
print("Data prepared")
 
#print total faces and labels
print("Total faces: ", len(faces))
print("Total labels: ", len(labels))	 

#create our LBPH face recognizer 
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
#face_recognizer = cv2.face.EigenFaceRecognizer_create()
#face_recognizer = cv2.face.FisherFaceRecognizer_create()
train_face_indentifier()


print("Predicting images...")
 
def prepare_test_data(dir_path):
	test_imgs = []
	images_names = os.listdir(dir_path)
	for image_name in images_names:	 
		if image_name.startswith("."): #ignore system files like .DS_Store
			continue;
		image_path = dir_path + "/" + image_name #build image path
		image = cv2.imread(image_path) #read image
		test_imgs.append(image)
	return test_imgs
#load test images
test_imgs = prepare_test_data("test_data")
 
#perform a prediction
predicted_imgs = []
for img in test_imgs:
	predicted_imgs.append(predict(img))
print("Prediction complete")
 
#display both images
for img in predicted_imgs:
	cv2.imshow("predict",img)	
	cv2.waitKey(0)
cv2.destroyAllWindows()