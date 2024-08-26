import cv2
import json
import pathlib, os
import tkinter as tk
from datetime import datetime

current_dir = pathlib.Path(__file__).parent
img_dir = current_dir.joinpath('img')
data_path = current_dir.joinpath('webcam.json')

if not img_dir.exists():
    os.makedirs(img_dir)

if data_path.exists():
    try:
        with open(data_path, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f'Gagal membuka file {data_path}. Coba hapus file tersebut.')
        exit()
else:
    data = []

def save_image(frame):
    root = tk.Tk()
    tk.Label(root, text='Masukkan nama').pack()
    name_entry = tk.Entry(root)
    name_entry.pack()

    def save():
        name = name_entry.get()
        timestamp = datetime.now().timestamp()
        img_path = img_dir.joinpath(f'{int(timestamp)}.jpg')
        cv2.imwrite(img_path, frame)

        data.append({
            'name': name,
            'path': str(img_path)
        })
        with open(data_path, 'w') as f:
            json.dump(data, f)
        root.destroy()

    tk.Button(text='Simpan', command=save).pack()

    root.mainloop()

cap = cv2.VideoCapture(0)

running = True
while running:
    frame = cap.read()[1]

    c = cv2.waitKey(1)
    if c == ord('q'):
        running = False
    elif c == ord('s'):
        save_image(frame)

    cv2.putText(frame, 'Tekan [s] untuk menyimpan, [q] untuk keluar', (0, 20), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow('Camera', frame)

cap.release()
cv2.destroyAllWindows()