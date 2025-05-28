import cv2
import threading
import tkinter as tk
from tkinter import Label, Button, ttk, messagebox
from PIL import Image, ImageTk
import time
from ultralytics import YOLO


class PersonCounterApp:
    def __init__(self, window):
        self.window = window
        self.window.configure(bg='#f0f0f0')
        
        self.model = None
        self.cap = None
        self.running = False
        
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0

        self.label = None
        self.count_label = None
        self.fps_label = None
        self.status_label = None
        self.start_button = None
        self.stop_button = None
        
        self.create_gui()
        self.load_model()

    def create_gui(self):
        title_frame = ttk.Frame(self.window)
        title_frame.pack(pady=25)
        
        title_label = tk.Label(
            title_frame, 
            text="YOLOv8n Kisi Sayimi",
            font=("Arial", 28, "bold"),
            bg='#f0f0f0',
            fg='#2c3e50'
        )
        title_label.pack()
        
        content_frame = ttk.Frame(self.window)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=30, pady=20)
        
        left_panel = ttk.Frame(content_frame, width=350)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 30))
        left_panel.pack_propagate(False)
        
        info_frame = ttk.LabelFrame(left_panel, text="Anlik Bilgiler", padding="15")
        info_frame.pack(fill=tk.X, pady=(0, 20))
        
        self.count_label = Label(
            info_frame, 
            text="Kisi Sayisi: 0", 
            font=("Arial", 22, "bold"),
            bg='#f0f0f0',
            fg='#e74c3c'
        )
        self.count_label.pack(pady=10)

        self.fps_label = Label(
            info_frame, 
            text="FPS: 0", 
            font=("Arial", 16),
            bg='#f0f0f0',
            fg='#3498db'
        )
        self.fps_label.pack(pady=5)

        self.status_label = Label(
            info_frame, 
            text="Model yukleniyor...", 
            font=("Arial", 14),
            bg='#f0f0f0',
            fg='#f39c12'
        )
        self.status_label.pack(pady=10)

        button_frame = ttk.LabelFrame(left_panel, text="Kamera Kontrolleri", padding="15")
        button_frame.pack(fill=tk.X, pady=(0, 20))

        self.start_button = Button(
            button_frame, 
            text="Kamerayi Baslat", 
            command=self.start_camera,
            font=("Arial", 16, "bold"),
            bg='#27ae60',
            fg='white',
            activebackground='#229954',
            activeforeground='white',
            padx=25,
            pady=15,
            relief='raised',
            bd=3,
            width=20
        )
        self.start_button.pack(pady=10, fill=tk.X)

        self.stop_button = Button(
            button_frame, 
            text="Kamerayi Durdur", 
            command=self.stop_camera,
            font=("Arial", 16, "bold"),
            bg='#e74c3c',
            fg='white',
            activebackground='#c0392b',
            activeforeground='white',
            padx=25,
            pady=15,
            relief='raised',
            bd=3,
            width=20,
            state='disabled'
        )
        self.stop_button.pack(pady=10, fill=tk.X)
        
        settings_frame = ttk.LabelFrame(left_panel, text="Ayarlar", padding="15")
        settings_frame.pack(fill=tk.X)
        
        conf_label = ttk.Label(settings_frame, text="Guven Esigi:", font=("Arial", 12))
        conf_label.pack(anchor='w', pady=(0, 5))
        
        self.conf_var = tk.DoubleVar(value=0.3)
        conf_scale = ttk.Scale(
            settings_frame, 
            from_=0.1, 
            to=0.9, 
            variable=self.conf_var,
            orient='horizontal',
            length=250
        )
        conf_scale.pack(fill=tk.X, pady=5)
        
        self.conf_value_label = ttk.Label(settings_frame, text="0.30", font=("Arial", 12, "bold"))
        self.conf_value_label.pack(pady=(5, 0))
        
        def update_conf_label(*args):
            self.conf_value_label.config(text=f"{self.conf_var.get():.2f}")
        
        self.conf_var.trace('w', update_conf_label)
        
        right_panel = ttk.Frame(content_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        video_frame = ttk.LabelFrame(right_panel, text="Kamera Goruntusu", padding="15")
        video_frame.pack(fill=tk.BOTH, expand=True)
        
        self.label = Label(
            video_frame, 
            text="Kamera goruntusu burada gorunecek\n\nKamerayi baslatmak icin sol paneldeki butona tiklayin",
            width=90,
            height=35,
            bg='black',
            fg='white',
            font=("Arial", 14),
            justify='center'
        )
        self.label.pack(padx=15, pady=15, fill=tk.BOTH, expand=True)

    def load_model(self):
        try:
            self.status_label.config(text="YOLO modeli yukleniyor...", fg='#f39c12')
            self.window.update()
            
            self.model = YOLO('yolov8n.pt')
            
            self.status_label.config(text="Model basariyla yuklendi", fg='#27ae60')
            print("Model basariyla yuklendi")
            
        except Exception as e:
            error_msg = f"Model yukleme hatasi: {str(e)}"
            self.status_label.config(text=error_msg, fg='#e74c3c')
            messagebox.showerror("Model Hatasi", error_msg)

    def start_camera(self):
        if not self.running and self.model:
            try:
                self.cap = cv2.VideoCapture(0)
                if not self.cap.isOpened():
                    raise Exception("Kamera acilamadi")
                
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                self.cap.set(cv2.CAP_PROP_FPS, 60)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
                self.running = True
                self.start_button.config(state='disabled')
                self.stop_button.config(state='normal')
                self.status_label.config(text="Kamera calisiyor", fg='#3498db')
                
                self.video_thread = threading.Thread(target=self.process_camera, daemon=True)
                self.video_thread.start()
                
                print("Kamera baslatildi")
                
            except Exception as e:
                error_msg = f"Kamera baslatma hatasi: {str(e)}"
                self.status_label.config(text=error_msg, fg='#e74c3c')
                messagebox.showerror("Kamera Hatasi", error_msg)

    def process_camera(self):
        while self.running and self.cap:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    break

                self.fps_counter += 1
                current_time = time.time()
                if current_time - self.fps_start_time >= 1.0:
                    self.current_fps = self.fps_counter
                    self.fps_counter = 0
                    self.fps_start_time = current_time
                    self.window.after(0, lambda: self.fps_label.config(text=f"FPS: {self.current_fps}"))

                confidence = self.conf_var.get()
                results = self.model.predict(
                    source=frame, 
                    conf=confidence, 
                    save=False, 
                    verbose=False,
                    classes=[0]
                )

                person_count = 0
                for result in results:
                    for box in result.boxes:
                        cls = int(box.cls[0])
                        if cls == 0:
                            person_count += 1
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            confidence_score = float(box.conf[0])
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                            label = f"Person {confidence_score:.2f}"
                            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                            cv2.rectangle(frame, (x1, y1-label_size[1]-15), 
                                          (x1+label_size[0]+10, y1), (0, 255, 0), -1)
                            cv2.putText(frame, label, (x1+5, y1-8), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

                total_text = f"Toplam Kisi: {person_count}"
                cv2.putText(frame, total_text, (15, 45), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                img = img.resize((960, 540), Image.Resampling.LANCZOS)
                imgtk = ImageTk.PhotoImage(image=img)

                self.window.after(0, lambda: self.update_gui(imgtk, person_count))

            except Exception as e:
                print(f"Frame isleme hatasi: {str(e)}")
                break

    def update_gui(self, imgtk, person_count):
        if self.running:
            self.label.imgtk = imgtk
            self.label.configure(image=imgtk, text="")
            self.count_label.config(text=f"Kisi Sayisi: {person_count}")

    def stop_camera(self):
        self.running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None

        self.label.config(
            image='', 
            text="Kamera durduruldu\n\nTekrar baslatmak icin sol paneldeki butona tiklayin"
        )
        self.count_label.config(text="Kisi Sayisi: 0")
        self.fps_label.config(text="FPS: 0")
        self.status_label.config(text="Kamera durduruldu", fg='#7f8c8d')
        self.start_button.config(state='normal')
        self.stop_button.config(state='disabled')
        print("Kamera durduruldu")

    def __del__(self):
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
