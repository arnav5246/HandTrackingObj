import cv2
import mediapipe as mp
import time  # To check the frame rate

cap = cv2.VideoCapture(0)  # Use the webcam

# Creating module to get certain points of hands
mpHands = mp.solutions.hands
hands = mpHands.Hands()  # 'Hands' is the parameter we are using and by default it is False.
#To draw line between each point in the hand using function of mediapipe
mpDraw = mp.solutions.drawing_utils

while True:
    success, img = cap.read()  # It will give the frame
    imgRGB = cv2.cvtColor9img, cv2.COLOR_BGR2RGB)  # To convert the image colour to RGB and hands uses only RGB.
    results = hands.process(imgRGB)  # It will process the hands function and yield the output.
    #print(results.multi_hand_landmarks) # To check if it detects any hand or not and as soon as it detects hand, it will print some output otherwise none.

    # We will use this for loop to check whether we have multiple hands or not
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:  # To extract information of each hand
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS) # 'mpHands.HAND_CONNECTIONS' will actually draw lines between all the points of the hand which is being detected.



    cv2.imshow("Image", img)  # Show the camera
    cv2.waitKey(1)
