import cv2
import face_recognition
import os
import numpy as np
from datetime import datetime

# Path ke folder dataset dan attendance
dataset_folder = 'dataset'
attendance_folder = 'attendance'

# Memuat data siswa
def load_known_faces():
    known_face_encodings = []
    known_face_names = []
    
    for filename in os.listdir(dataset_folder):
        if filename.endswith('.jpg'):
            image_path = os.path.join(dataset_folder, filename)
            image = face_recognition.load_image_file(image_path)
            encoding = face_recognition.face_encodings(image)[0]
            name = filename.split('.')[0]
            
            known_face_encodings.append(encoding)
            known_face_names.append(name)
    
    return known_face_encodings, known_face_names

# Menulis catatan absensi
def mark_attendance(name, status):
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d %H:%M:%S")
    attendance_file = os.path.join(attendance_folder, 'attendance.csv')
    
    if not os.path.isfile(attendance_file):
        with open(attendance_file, 'w') as f:
            f.write('Name,Date,Status\n')
    
    with open(attendance_file, 'a') as f:
        f.write(f'{name},{dt_string},{status}\n')

# Main function
def main():
    known_face_encodings, known_face_names = load_known_faces()
    
    video_capture = cv2.VideoCapture(0)
    
    while True:
        ret, frame = video_capture.read()
        rgb_frame = frame[:, :, ::-1]
        
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]
            
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)
            
            if name != "Unknown":
                mark_attendance(name, 'Arrived')
            else:
                cv2.putText(frame, 'Press s to register', (10, 30), font, 0.75, (0, 0, 255), 2)
        
        cv2.imshow('Video', frame)
        
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
        elif key & 0xFF == ord('s'):
            # Tangkap dan simpan gambar untuk pendaftaran
            cv2.imwrite(f'{dataset_folder}/unknown_{datetime.now().strftime("%Y%m%d%H%M%S")}.jpg', frame)
    
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)
    if not os.path.exists(attendance_folder):
        os.makedirs(attendance_folder)
    main()
