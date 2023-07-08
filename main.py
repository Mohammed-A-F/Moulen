import cv2
import dlib
import numpy as np
from imutils import face_utils
import time
import streamlit as st

# Load gender and age prediction models
gender_model = cv2.dnn.readNetFromCaffe('mods/gender_deploy.prototxt', 'mods/gender_net.caffemodel')
age_model = cv2.dnn.readNetFromCaffe('mods/age_deploy.prototxt', 'mods/age_net.caffemodel')

# Set confidence thresholds for gender and age predictions
gender_confidence_threshold = 0.5
age_confidence_threshold = 0.9

# Load face detector and shape predictor from dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('mods/shape_predictor_68_face_landmarks.dat')

# Open the video capture device
cap = cv2.VideoCapture(0)

# Initialize variables
total_faces = 0
total_age = 0
total_male = 0
total_female = 0
start_time = time.time()

# Define CSS style for the page
page_bg = """
<style>
body {
background-color: #F5F5F5;
}
</style>
"""

# Write the CSS style to the page
st.markdown(page_bg, unsafe_allow_html=True)

# Define HTML template for displaying age and gender predictions
template = """
<div style='padding: 10px; border: 1px solid #ccc; border-radius: 5px;'>
    <div style='display: flex; justify-content: space-between;'>
        <div><b>Gender:</b> {}</div>
        <div><b>Age:</b> {}</div>
    </div>
    <div style='margin-top: 10px;'>{}: {:.2f}%</div>
</div>
"""

# Define HTML template for displaying the video
video_template = """
<div style='padding: 10px; border: 1px solid #ccc; border-radius: 5px;'>
    <video width='100%' controls>
        <source src='{}' type='video/mp4'>
    </video>
</div>
"""

# Loop over frames from the video capture device
# Initialize the counter
counter = 0

# Loop over frames from the video capture device
while True:
    # Increment the counter
    counter += 1

    # Break out of the loop if the counter reaches 3
    if counter > 1:
        break

    # Read a frame from the video capture device
    ret, frame = cap.read()

    # If the frame cannot be read, continue to the next iteration of the loop
    if not ret:
        continue

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    rects = detector(gray, 0)

    # Loop over the face detections
    for rect in rects:
        # Get the bounding box coordinates for the face
        (x, y, w, h) = face_utils.rect_to_bb(rect)

        # Extract the face ROI
        face = frame[y:y + h, x:x + w]

        # If the face ROI is empty, continue to the next iteration of the loop
        if face.size == 0:
            continue

        # Prepare the face ROI for gender and age prediction
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)

        # Make gender and age predictions on the face ROI
        gender_model.setInput(blob)
        gender_predictions = gender_model.forward()
        age_model.setInput(blob)
        age_predictions = age_model.forward()

        # Determine the gender and age
        gender = 'Male' if gender_predictions[0][0] > gender_predictions[0][1] else 'Female'
        age_index = np.argmax(age_predictions)
        age = age_index + 1

        # Display the gender and age predictions on the frame
        text = '{}: {:.2f}%'.format(gender, max(gender_predictions[0]) * 100)
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        age = int(age * 3.8)
        text = 'Age: {}'.format(age)
        cv2.putText(frame, text, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Update the total number of faces, total age, total male, and total female
        total_faces += 1
        total_age += age
        if gender == 'Male':
            total_male += 1
        else:
            total_female += 1

        # Display the age and gender predictions using the HTML template
        st.markdown(template.format(gender, age, gender, max(gender_predictions[0]) * 100), unsafe_allow_html=True)

    # Display the processed frame using the st.image() component
    st.image(frame, channels="BGR")

    # Check if it is time to display the video
    if time.time() - start_time > 3:
        if total_faces > 0:
            # Calculate the average age, male ratio, and female ratio
            avg_age = total_age / total_faces
            avg_male = total_male / total_faces
            avg_female = total_female / total_faces

            # Choose the video to display based on the average age and gender ratio
            if avg_age <= 12:
                if avg_male > avg_female:
                    st.video('videos/kids_male_video.mp4')
                else:
                    st.video('videos/kids_female_video.mp4')
            else:
                if avg_male > avg_female:
                    st.video('videos/adult_male_video.mp4')
                else:
                    st.video('videos/adult_female_video.mp4')




            # Open the video file
            video = cv2.VideoCapture(video_path)

            # Loop over the frames in the video file and display them using the st.video() component
            while True:
                ret, frame = video.read()
                st.video(video_path)

                if not ret:
                    break


                # Convert the frame to RGB format
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Display the frame using the st.video() component
                st.markdown(video_template.format(video_path), unsafe_allow_html=True)


            # Release the video file
            video.release()

        # Reset the variables
        total_faces = 0
        total_age = 0
        total_male = 0
        total_female = 0
        start_time = time.time()

st.video('videos/adult_female_video.mp4')
