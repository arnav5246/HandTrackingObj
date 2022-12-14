import cv2
import mediapipe as mp
import time  # To check the frame rate

cap = cv2.VideoCapture(0)  # Use the webcam

# Creating module to get certain points of hands
mpHands = mp.solutions.hands
hands = mpHands.Hands()  # 'Hands' is the parameter we are using and by default it is False.
#To draw line between each point in the hand using function of mediapipe
mpDraw = mp.solutions.drawing_utils

#To showcase Frame Rate
pTime = 0 # Previous time
cTime = 0 # Current time
while True:
    success, img = cap.read()  # It will give the frame
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # To convert the image colour to RGB and hands uses only RGB.
    results = hands.process(imgRGB)  # It will process the hands function and yield the output.
    #print(results.multi_hand_landmarks) # To check if it detects any hand or not and as soon as it detects hand, it will print some output otherwise none.

    # We will use this for loop to check whether we have multiple hands or not
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:  # To extract information of each hand
            for id, lm in enumerate(handLms.landmark): # lm gives landmark and id gives index number to each finger
                #print(id,lm)
                h, w, c = img.shape # determine height, width and chance
                cx, cy = int(lm.x*w), int(lm.y*h) # cx and cy are the positions
                print(id, cx, cy)
                # To determine or highlight specific landmark using cx and cy coordinates
                if id == 4: # specifying id number
                    cv2.circle(img, (cx,cy), 15, (255,0,255), cv2.FILLED) # circle out the point with given coordinates and size 15 and colour


            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS) # 'mpHands.HAND_CONNECTIONS' will actually draw lines between all the points of the hand which is being detected.

    cTime = time.time() # Display fps
    fps = 1/(cTime-pTime) # Assign fps
    pTime = cTime # previous time is now current time

    # we want text of "img", converted fps to integer, gave position of fps "10,70", gave fps a font "Hershey..", gave the scale "3", gave colour "Purple 255,0,255", gave the thickness as "3"
    cv2.putText(img,str(int(fps)),(10,70), cv2.FONT_HERSHEY_PLAIN,3,
              (255,0,255),3)


    cv2.imshow("Image", img)  # Show the camera
    cv2.waitKey(1)