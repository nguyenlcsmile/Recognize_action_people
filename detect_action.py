import cv2
import mediapipe as mp
import numpy as np
from model_utils import define_model, load_model

mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results

def draw_styled_landmarks(image, results):
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=2), 
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=1)
                             ) 

def detect_action_webcam():
	model = define_model()
	model = load_model(model)
	print("[INFO] Model is loaded ...")
	class_name = ["kick", "punch", "squat", "stand", "wave"]
	
	cap = cv2.VideoCapture("test.mp4")
	cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
	cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
	with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
		while cap.isOpened():
			ret, frame = cap.read()
			if ret:
				try:
					# Make detections
					image, results = mediapipe_detection(frame, holistic)
					# Extract landmarks
					pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)
					print(len(pose))
					pose = np.expand_dims(pose, axis=0)
					res = model.predict(pose)
					action = class_name[np.argmax(res)]
					# Draw landmarks
					draw_styled_landmarks(image, results)
					cv2.putText(image, action, (int(50),int(50)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,255), 2)
					# Show to screen
					cv2.imshow('Realtime action', cv2.resize(image, (640,480)))
					# Break gracefully
					if cv2.waitKey(10) & 0xFF == ord('q'):
						break
				except:
					pass
			else:
				break 
		cap.release()
		cv2.destroyAllWindows()


