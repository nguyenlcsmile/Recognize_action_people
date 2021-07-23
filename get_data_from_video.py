import cv2
import time
import os
import pathlib
import pandas as pd 
import numpy as np 
import mediapipe as mp 

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

def main():
    id_class = -1
    df = pd.DataFrame()
    path_folder = pathlib.Path('video')
    video_paths = sorted(list(path_folder.glob("*.gif")))
    print(video_paths)
    for path in video_paths:
        datasets = []
        id_class += 1
        # print(str(path).split("\\")[1].split('.')[0].split('0')[0])
        cap = cv2.VideoCapture('{}'.format(path))
        time.sleep(1)
        if cap is None or not cap.isOpened():
            print('Khong the mo file video')
            return
        cv2.namedWindow('Generate Data', cv2.WINDOW_AUTOSIZE)
        n = 1
        dem = 1
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            while True:
                [success, frame] = cap.read()
                ch = cv2.waitKey(30)
                if success:
                    # Make detections
                    image, results = mediapipe_detection(frame, holistic)
                    #print(results)
            
                    # Extract landmarks
                    pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)
                    datasets.append(pose)
                    # Draw landmarks
                    draw_styled_landmarks(image, results)
                    cv2.imshow('Image', image)

                    # Break gracefully
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break
                else:
                    break
        df_ = pd.DataFrame(data=datasets)
        df_['class_id'] = id_class
        df = df.append(df_, ignore_index=True)
    # print(df)
    df.to_csv('data_raw.csv', index=False)
    return df 
    
if __name__ == "__main__":
    main()
