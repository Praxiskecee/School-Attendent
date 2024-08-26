import cv2
import json
import pathlib
import os
import sqlite3
import tkinter as tk
from datetime import datetime, date, timedelta
from PIL import Image, ImageTk
import numpy as np
import face_recognition

# Paths
current_dir = pathlib.Path(__file__).parent
img_dir = current_dir.joinpath('img')
data_path = current_dir.joinpath('webcam.json')
db_path = current_dir.joinpath('faces.db')

# Create directories if not exist
img_dir.mkdir(exist_ok=True)

# Load existing data if available
if data_path.exists():
    try:
        with open(data_path, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f'Gagal membuka file {data_path}. Coba hapus file tersebut.')
        exit()
else:
    data = []

# Connect to SQLite database
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Create table for storing face data
cursor.execute('''
CREATE TABLE IF NOT EXISTS faces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    role TEXT,
    image_path TEXT,
    encoding BLOB,
    log TEXT
)
''')
conn.commit()

# Dictionary to store the last detected time for each face
last_detected_times = {}

def recognize_face(face_encoding):
    cursor.execute("SELECT * FROM faces")
    rows = cursor.fetchall()
    for row in rows:
        known_encoding = np.frombuffer(row[4], dtype=np.float64)
        matches = face_recognition.compare_faces([known_encoding], face_encoding, tolerance=0.6)
        if True in matches:
            return row
    return None

def save_identity(name, face, role, face_encoding):
    timestamp = datetime.now().timestamp()
    img_path = img_dir.joinpath(f'{int(timestamp)}.jpg')
    cv2.imwrite(str(img_path), face)

    encoding_blob = face_encoding.tobytes()
    cursor.execute('''
    INSERT INTO faces (name, role, image_path, encoding, log) VALUES (?, ?, ?, ?, ?)
    ''', (name, role, str(img_path), encoding_blob, '[]'))
    conn.commit()

def register_new_face(frame, face, face_encoding):
    root = tk.Tk()
    root.title("Daftarkan Wajah")

    tk.Label(root, text='Masukkan nama:').pack(padx=10, pady=10)
    name_entry = tk.Entry(root)
    name_entry.pack(padx=10, pady=10)

    tk.Label(root, text='Masukkan role:').pack(padx=10, pady=10)
    role_entry = tk.Entry(root)
    role_entry.pack(padx=10, pady=10)

    # Show the face to be saved
    face_img = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
    face_img = ImageTk.PhotoImage(image=face_img)
    tk.Label(root, image=face_img).pack()

    def save():
        name = name_entry.get().strip()
        role = role_entry.get().strip()
        if name and role:
            save_identity(name, face, role, face_encoding)
            root.destroy()

    tk.Button(root, text='Simpan', command=save).pack(pady=10)
    root.mainloop()

def log_attendance(entry_id):
    today = date.today().isoformat()
    now = datetime.now().isoformat()

    cursor.execute("SELECT log FROM faces WHERE id = ?", (entry_id,))
    log = eval(cursor.fetchone()[0])

    if len(log) == 0 or log[-1]['date'] != today:
        # Jika belum ada log untuk hari ini, tambahkan log kehadiran
        log.append({
            'date': today,
            'attendance_time': now,
            'go_home_time': None
        })
        print(f"{datetime.now()}: {entry_id} - Kedatangan tercatat pada {now}")
    else:
        # Periksa apakah waktu pulang sudah dicatat
        if log[-1]['go_home_time'] is None:
            log[-1]['go_home_time'] = now
            print(f"{datetime.now()}: {entry_id} - Pulang tercatat pada {now}")
        else:
            # Jika waktu pulang sudah dicatat, berarti ini adalah kedatangan baru di hari yang sama
            log.append({
                'date': today,
                'attendance_time': now,
                'go_home_time': None
            })
            print(f"{datetime.now()}: {entry_id} - Kedatangan tercatat pada {now}")

    cursor.execute("UPDATE faces SET log = ? WHERE id = ?", (str(log), entry_id))
    conn.commit()

def process_video():
    cap = cv2.VideoCapture(0)

    detected_faces = set()
    unknown_detected = False
    unknown_face = None
    face_encoding_for_register = None

    running = True
    while running:
        ret, frame = cap.read()
        if not ret:
            print("Gagal membaca frame dari webcam.")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            face_img = frame[top:bottom, left:right]
            face_key = (left, top, right, bottom)

            if face_key not in detected_faces:
                entry = recognize_face(face_encoding)
                detected_faces.add(face_key)
                current_time = datetime.now()

                if entry:
                    # Check if the face was detected recently
                    last_detected_time = last_detected_times.get(entry[0], None)
                    if last_detected_time and (current_time - last_detected_time) < timedelta(minutes=5):
                        # Face was detected recently, skip logging
                        continue
                    last_detected_times[entry[0]] = current_time
                    log_attendance(entry[0])
                    name_and_role = f"{entry[1]} ({entry[2]})"
                    cv2.putText(frame, name_and_role, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                else:
                    # Jika wajah tidak dikenal, beri label "Unknown"
                    cv2.putText(frame, 'Unknown', (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    unknown_detected = True
                    unknown_face = face_img
                    face_encoding_for_register = face_encoding

                cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)

        # Tampilkan teks untuk mendaftarkan wajah di bagian atas frame kamera
        if unknown_detected:
            cv2.putText(frame, 'Tekan [r] untuk mendaftarkan wajah ini', (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.putText(frame, 'Tekan [q] untuk keluar', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow('Camera', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            running = False
        elif key == ord('r') and unknown_detected:
            register_new_face(frame, unknown_face, face_encoding_for_register)
            unknown_detected = False

    cap.release()
    cv2.destroyAllWindows()

# Run the video processing
process_video()

# Close database connection after processing
conn.close()
