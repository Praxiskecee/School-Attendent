import cv2
import face_recognition
import os
import numpy as np
from datetime import datetime

# Folder untuk menyimpan data wajah dan nama
faces_folder = "registered_faces"

# Pastikan folder terdaftar ada
if not os.path.exists(faces_folder):
    os.makedirs(faces_folder)

# Fungsi untuk menyimpan foto wajah dan nama pengguna
def save_registration(name, image):
    # Simpan gambar wajah
    file_path = os.path.join(faces_folder, f"{name}.jpg")
    cv2.imwrite(file_path, image)

    print(f"Wajah {name} telah disimpan di {file_path}")

# Fungsi untuk memuat semua wajah yang telah terdaftar
def load_registered_faces():
    known_faces = []
    known_face_encodings = []
    
    for file_name in os.listdir(faces_folder):
        if file_name.endswith(".jpg"):
            name = os.path.splitext(file_name)[0]
            image = face_recognition.load_image_file(os.path.join(faces_folder, file_name))
            encoding = face_recognition.face_encodings(image)[0]
            
            known_faces.append(name)
            known_face_encodings.append(encoding)
    
    return known_faces, known_face_encodings

# Fungsi utama untuk menjalankan sistem
def start_system():
    # Ambil referensi ke webcam
    video_capture = cv2.VideoCapture(0)
    
    # Muat wajah yang telah terdaftar
    known_faces, known_face_encodings = load_registered_faces()
    
    process_this_frame = True
    
    while True:
        ret, frame = video_capture.read()
        
        # Hanya proses setiap frame kedua untuk menghemat waktu
        if process_this_frame:
            # Ubah ukuran frame video ke 1/4 ukuran asli untuk mempercepat pemrosesan
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            # Temukan semua wajah dan encoding wajah dalam frame video saat ini
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            
            for face_encoding in face_encodings:
                # Lihat apakah wajah cocok dengan wajah yang dikenal
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"
                
                if True not in matches:
                    # Jika wajah tidak dikenal, lakukan pendaftaran
                    print("Wajah baru terdeteksi. Silakan registrasi.")
                    name = input("Masukkan nama Anda: ")
                    save_registration(name, frame)
                    
                    # Tambahkan wajah yang baru terdaftar ke sistem
                    known_faces.append(name)
                    known_face_encodings.append(face_encoding)
                    
                    print(f"{name} telah terdaftar.")
                else:
                    # Temukan wajah yang paling cocok
                    best_match_index = np.argmin(face_recognition.face_distance(known_face_encodings, face_encoding))
                    if matches[best_match_index]:
                        name = known_faces[best_match_index]
                        print(f"Wajah dikenali: {name}")
                
                # Tampilkan nama yang dikenali pada frame
                for (top, right, bottom, left) in face_locations:
                    top *= 4
                    right *= 4
                    bottom *= 4
                    left *= 4
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        
        process_this_frame = not process_this_frame
        
        # Tampilkan frame yang dihasilkan
        cv2.imshow('Video', frame)
        
        # Tekan 'q' untuk keluar dari loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Lepaskan kendali atas webcam
    video_capture.release()
    cv2.destroyAllWindows()

# Mulai sistem
start_system()
