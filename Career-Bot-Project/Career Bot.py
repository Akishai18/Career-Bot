import os
import cv2
import math
import numpy as np
import google.generativeai as genai
import speech_recognition as sr
import pyttsx3
import threading
import time

genai.configure(api_key="AIzaSyBWaI02BmbxTzpD2RqQeMIMiAP4XVG90P4")

# Set up the model
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 0,
    "max_output_tokens": 8192,
}

safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
]

model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest",
                              generation_config=generation_config,
                              safety_settings=safety_settings)

convo = model.start_chat(history=[
])

r = sr.Recognizer()

#Function converts written text into speech using pyttsx3 module
def SpeakText(command):

    engine = pyttsx3.init()

    engine.say(command)

    engine.runAndWait()

def calculate_center(x, y, w, h):
    #calculate the coordinate of the centre of the eye
    return (x + w // 2, y + h // 2)

def calculate_angle(left_eye_center, right_eye_center):
    #Find the change of the x and y coordinates for the eye centres
    delta_y = right_eye_center[1] - left_eye_center[1]
    delta_x = right_eye_center[0] - left_eye_center[0]
    #Calculates the angle of eyes by finding the arctangent of the new points
    angle = math.atan2(delta_y, delta_x) * 180 / math.pi
    return angle

def is_looking_at_camera(eye_centers):
    angle_limit=15

    if len(eye_centers) != 2:
        return False

    left_eye_center = eye_centers[0]
    right_eye_center = eye_centers[1]

    # Calculate the angle between the eyes
    angle = calculate_angle(left_eye_center, right_eye_center)
    #It is important to note that there are multiple ways to check if someone is looking at a specifc point with variable accuracy. I have chosen to use angles as 
    #after a lot of trial and error with other methods this provided the best results 

    # Check if the angle is within the threshold for looking straight
    if abs(angle) < angle_limit:
        return True
    else:
        return False

def start_camera():
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Creates the path to the Haarcascade files
    face_cascade_path = os.path.join(current_dir, 'haarcascade_frontalface_default.xml')
    eye_cascade_path = os.path.join(current_dir, 'haarcascade_eye_tree_eyeglasses.xml')

    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    eye_cascade = cv2.CascadeClassifier(eye_cascade_path)

    # Initialize the webcam
    cap = cv2.VideoCapture(0)

    #Create an infinite loop that captures frames through the webcam
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break
        
        #Use the cvtColor function to convert the frame from color into grayscale to make the analysis less complex
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #Uses the face cascade pre - trained classifiers to detect faces in the image 
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)

            roi_gray = gray[y:y + h, x:x + w]

            roi_color = frame[y:y + h, x:x + w]
            #Detects eyes in the region of interest (face) using the haarcascade pre trained classifier
            eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=10, minSize=(15, 15))

            #define an empty list that will store coordinates of the centre of the left and right eyes
            eye_centers = []
            #iterate over a loop for each eye with loop variables corresponding to rectangular coordinates for the eyes
            for (ex, ey, ew, eh) in eyes:

                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

                eye_center = calculate_center(ex, ey, ew, eh)

                eye_centers.append(eye_center)

                cv2.circle(roi_color, eye_center, 5, (0, 255, 255), -1)

            #call the function to check if the user is looking at the camera
            if is_looking_at_camera(eye_centers):
                cv2.putText(frame, "Looking at Camera", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                cv2.putText(frame, "Not looking at Camera", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        # Display the frames
        cv2.imshow('Video', frame)
        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()

def speech_to_text():
    while True:
        try:
            # Use the microphone as source for input.
            with sr.Microphone() as source2:
                r.adjust_for_ambient_noise(source2, duration=0.2)

                # Listen for the user's input
                audio2 = r.listen(source2)

                # Use google to recognize audio
                MyText = r.recognize_google(audio2)
                MyText = MyText.lower()

                print("Answer Received, Processing: ")
                #Send the message to the Google Gemini program through an API request 
                convo.send_message("Provide Feedback to the Question you gave me based on my answer: Please DO NOT WRITE ASTERICKS {}".format(MyText))

                #The bot speaks the response and it is printed as well
                SpeakText(convo.last.text)
                print(convo.last.text)
                break

        except sr.RequestError:
            print("Could not request results; {0}".format(sr.RequestError))

        except sr.UnknownValueError:
            print("unknown error occurred")

def Interview():
    while True:
        question = input("What type of job are you applying for: ")
        type = input("What type of question would you like to practice (behavioural, technical, practical): ")
        convo.send_message("Generate me a {} question relating to {}: ".format(type, question))
        print(convo.last.text)
        command = input("When you are ready please enter ready, if you would like to exit please enter exit: ")
        if command == "exit":
            break
        elif command == "ready":
            # Start the camera feed and speech recognition simultaneously using threading
            camera_thread = threading.Thread(target=start_camera)
            speech_thread = threading.Thread(target=speech_to_text)
            camera_thread.start()
            speech_thread.start()

            camera_thread.join()
            speech_thread.join()


def resume():
    command = input("What would you like help with your resume on: ")
    convo.send_message("Based on this statement provide feedback for the resume {}".format(command))
    print(convo.last.text)

def CL():
    command = input("What would you like help with your cover letter on: ")
    convo.send_message("Based on this statement provide feedback for the cover letter {}".format(command))
    print(convo.last.text)

print("Hello Welcome to Career Bot ")

while True:
    choice = input("To discuss and improve your resume please enter Resume:\nTo discuss and improve your cover letter please enter Cover Letter\nTo practice interviewing please enter Interview: ")

    if choice == "Resume":
        resume()
    elif choice == "Cover Letter":
        CL()
    elif choice == "Interview":
        Interview()
    elif choice == "Exit":
        break
    else:
        print("Please Try Again")
