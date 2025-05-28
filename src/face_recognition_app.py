import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk
import threading
import os
import pickle
import glob
import time

# InsightFace ve sklearn import'ları
try:
    from insightface import app as face_app
    from sklearn.metrics.pairwise import cosine_similarity
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False

# Performance ayarları
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.configure(bg='#f0f0f0')
        
        # Değişkenler
        self.camera = None
        self.is_running = False
        self.face_app = None
        self.known_embeddings = {}
        self.similarity_threshold = 0.4
        self.frame_skip = 2
        self.frame_counter = 0
        self.last_faces = []
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
        # Veri yolları
        self.data_path = "Enter your data folder path here"
        self.embeddings_path = "embeddings/face_embeddings.pkl"
        
        # GUI elementleri
        self.video_frame = None
        self.fps_label = None
        self.status_label = None
        self.start_button = None
        self.threshold_var = None
        
        # InsightFace kontrolü
        if not INSIGHTFACE_AVAILABLE:
            messagebox.showerror(
                "Eksik Kutuphane", 
                "InsightFace kutuphanesi bulunamadi!\nLutfen su komutu calistirin:\npip install insightface"
            )
            return
            
        # GUI oluştur
        self.create_gui()
        
        # Modelleri yükle
        self.load_models()
        
        # Kayıtlı yüzleri yükle veya oluştur
        self.load_or_create_embeddings()
    
    def create_gui(self):
        """Yüz tanıma arayüzünü oluştur"""
        # Ana başlık
        title_frame = ttk.Frame(self.root)
        title_frame.pack(pady=20)
        
        title_label = tk.Label(
            title_frame, 
            text="AI Tabanli Yuz Tanima",
            font=("Arial", 20, "bold"),
            bg='#f0f0f0',
            fg='#2c3e50'
        )
        title_label.pack()
        
        # Ana içerik frame
        content_frame = ttk.Frame(self.root)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=20)
        
        # Sol panel - Kontroller
        left_panel = ttk.Frame(content_frame, width=250)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 20))
        left_panel.pack_propagate(False)
        
        # Kamera kontrolleri
        camera_frame = ttk.LabelFrame(left_panel, text="Kamera Kontrolleri", padding="10")
        camera_frame.pack(fill=tk.X, pady=(0, 20))
        
        self.start_button = tk.Button(
            camera_frame, 
            text="Kamerayi Baslat", 
            command=self.toggle_camera,
            font=("Arial", 12, "bold"),
            bg='#27ae60',
            fg='white',
            activebackground='#229954',
            activeforeground='white',
            padx=15,
            pady=8,
            relief='raised',
            bd=2,
            width=20
        )
        self.start_button.pack(pady=5)
        
        # Ayarlar
        settings_frame = ttk.LabelFrame(left_panel, text="Ayarlar", padding="10")
        settings_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Benzerlik eşiği
        ttk.Label(settings_frame, text="Tanima Esigi:").pack(anchor='w')
        
        self.threshold_var = tk.DoubleVar(value=0.4)
        threshold_scale = ttk.Scale(
            settings_frame, 
            from_=0.1, 
            to=0.8, 
            variable=self.threshold_var,
            orient='horizontal'
        )
        threshold_scale.pack(fill=tk.X, pady=5)
        
        self.threshold_label = ttk.Label(settings_frame, text="0.40")
        self.threshold_label.pack()
        
        def update_threshold(*args):
            value = self.threshold_var.get()
            self.threshold_label.config(text=f"{value:.2f}")
            self.similarity_threshold = value
        
        self.threshold_var.trace('w', update_threshold)
        
        # Veri yönetimi
        data_frame = ttk.LabelFrame(left_panel, text="Veri Yonetimi", padding="10")
        data_frame.pack(fill=tk.X, pady=(0, 20))
        
        ttk.Button(
            data_frame, 
            text="Data Klasoru Sec",
            command=self.select_data_folder
        ).pack(fill=tk.X, pady=2)
        
        ttk.Button(
            data_frame, 
            text="Embeddings Yenile",
            command=self.refresh_embeddings
        ).pack(fill=tk.X, pady=2)
        
        # Durum bilgileri
        status_frame = ttk.LabelFrame(left_panel, text="Durum", padding="10")
        status_frame.pack(fill=tk.X)
        
        self.fps_label = ttk.Label(status_frame, text="FPS: 0", foreground="blue")
        self.fps_label.pack(anchor='w', pady=2)
        
        self.status_label = ttk.Label(status_frame, text="Hazir", foreground="green")
        self.status_label.pack(anchor='w', pady=2)
        
        self.persons_label = ttk.Label(status_frame, text="Kayitli: 0 kisi", foreground="purple")
        self.persons_label.pack(anchor='w', pady=2)
        
        # Sağ panel - Video görüntüsü
        right_panel = ttk.Frame(content_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Video frame
        video_container = ttk.Frame(right_panel, relief='solid', borderwidth=2)
        video_container.pack(fill=tk.BOTH, expand=True)
        
        self.video_frame = tk.Label(
            video_container, 
            text="Kamera goruntusu burada gorunecek\n\nYuz tanima icin kamerayi baslatin",
            background="black", 
            foreground="white",
            font=("Arial", 14),
            justify='center'
        )
        self.video_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    def load_models(self):
        """InsightFace modellerini yükle - GPU optimizasyonlu"""
        try:
            self.status_label.config(text="Modeller yukleniyor (GPU)...", foreground="orange")
            self.root.update()
            
            # GPU provider'ları sırasıyla dene
            providers = [
                'CUDAExecutionProvider',  # NVIDIA GPU
                'CPUExecutionProvider'    # Fallback
            ]
            
            # InsightFace uygulamasını başlat
            self.face_app = face_app.FaceAnalysis(providers=providers)
            self.face_app.prepare(ctx_id=0, det_size=(320, 320))  # Daha küçük boyut = daha hızlı
            
            # Kullanılan provider'ı kontrol et
            active_provider = self.face_app.models['detection'].session.get_providers()[0]
            provider_text = "GPU" if "CUDA" in active_provider else "CPU"
            
            self.status_label.config(text=f"Modeller yuklendi ({provider_text})", foreground="green")
            print(f"Modeller basariyla yuklendi - Provider: {active_provider}")
            
        except Exception as e:
            error_msg = f"Model yukleme hatasi: {str(e)}"
            self.status_label.config(text="Model yukleme hatasi", foreground="red")
            messagebox.showerror("Hata", error_msg)
    
    def load_or_create_embeddings(self):
        """Kayıtlı yüz vektörlerini yükle veya oluştur"""
        try:
            # Embeddings klasörünü oluştur
            os.makedirs("embeddings", exist_ok=True)
            
            if os.path.exists(self.embeddings_path):
                # Mevcut embeddings'leri yükle
                with open(self.embeddings_path, 'rb') as f:
                    self.known_embeddings = pickle.load(f)
                print(f"Kayitli {len(self.known_embeddings)} kisi yuklendi")
                self.persons_label.config(text=f"Kayitli: {len(self.known_embeddings)} kisi")
            else:
                # Yeni embeddings oluştur
                self.create_embeddings_from_data()
                
        except Exception as e:
            print(f"Embeddings yukleme hatasi: {str(e)}")
            self.known_embeddings = {}
            self.persons_label.config(text="Kayitli: 0 kisi")
    
    def create_embeddings_from_data(self):
        """Veri klasöründeki resimlerden yüz vektörleri oluştur"""
        if not self.face_app:
            print("Face app yuklenmedi!")
            return
            
        self.status_label.config(text="Yuz vektorleri olusturuluyor...", foreground="orange")
        self.root.update()
        
        if not os.path.exists(self.data_path):
            print(f"Data klasoru bulunamadi: {self.data_path}")
            self.status_label.config(text="Data klasoru bulunamadi", foreground="red")
            return
        
        # Alt klasörlerdeki kişi isimlerini al
        people_dirs = [d for d in os.listdir(self.data_path) 
                      if os.path.isdir(os.path.join(self.data_path, d))]
        
        if not people_dirs:
            print("Data klasorunde kisi klasorleri bulunamadi")
            self.status_label.config(text="Kisi klasorleri bulunamadi", foreground="red")
            return
        
        processed_people = 0
        
        for person_name in people_dirs:
            person_path = os.path.join(self.data_path, person_name)
            embeddings_list = []
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
            
            for ext in image_extensions:
                image_files = glob.glob(os.path.join(person_path, ext))
                image_files.extend(glob.glob(os.path.join(person_path, ext.upper())))
                
                for img_path in image_files:
                    try:
                        # Resmi oku
                        img = cv2.imread(img_path)
                        if img is None:
                            continue
                            
                        # Yüz tespit et
                        faces = self.face_app.get(img)
                        
                        if len(faces) > 0:
                            # En büyük yüzü al
                            face = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
                            embeddings_list.append(face.embedding)
                            
                    except Exception as e:
                        print(f"Resim isleme hatasi {img_path}: {str(e)}")
                        continue
            
            if embeddings_list:
                # Ortalama embedding hesapla
                avg_embedding = np.mean(embeddings_list, axis=0)
                self.known_embeddings[person_name] = avg_embedding
                processed_people += 1
                print(f"{person_name}: {len(embeddings_list)} resimden embedding olusturuldu")
        
        if processed_people > 0:
            # Embeddings'leri kaydet
            with open(self.embeddings_path, 'wb') as f:
                pickle.dump(self.known_embeddings, f)
                
            self.status_label.config(text="Yuz vektorleri hazir", foreground="green")
            self.persons_label.config(text=f"Kayitli: {len(self.known_embeddings)} kisi")
            print(f"Toplam {len(self.known_embeddings)} kisi kaydedildi")
        else:
            self.status_label.config(text="Hic yuz bulunamadi", foreground="red")
    
    def recognize_face(self, face_embedding):
        """Yüz tanıma işlemi"""
        if not self.known_embeddings:
            return "Unknown", 0.0
            
        best_match = "Unknown"
        best_similarity = 0.0
        
        for name, known_embedding in self.known_embeddings.items():
            # Cosine similarity hesapla
            similarity = cosine_similarity([face_embedding], [known_embedding])[0][0]
            
            if similarity > best_similarity and similarity > self.similarity_threshold:
                best_similarity = similarity
                best_match = name
                
        return best_match, best_similarity
    
    def select_data_folder(self):
        """Data klasörü seçme"""
        folder_path = filedialog.askdirectory(
            title="Yuz verisi klasorunu secin",
            initialdir=os.path.dirname(self.data_path) if os.path.exists(self.data_path) else os.getcwd()
        )
        
        if folder_path:
            self.data_path = folder_path
            print(f"Yeni data klasoru: {folder_path}")
            self.refresh_embeddings()
    
    def refresh_embeddings(self):
        """Embeddings'leri yeniden oluştur"""
        self.known_embeddings = {}
        self.create_embeddings_from_data()
    
    def toggle_camera(self):
        """Kamerayı başlat/durdur"""
        if not self.is_running:
            self.start_camera()
        else:
            self.stop_camera()
    
    def start_camera(self):
        """Kamerayı başlat - GPU optimizasyonlu"""
        try:
            self.camera = cv2.VideoCapture(0)
            if not self.camera.isOpened():
                raise Exception("Kamera acilamadi")
            
            # Kamera ayarları - performans için optimize edildi
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.camera.set(cv2.CAP_PROP_FPS, 30)
            self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Buffer boyutunu küçült
            self.camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))  # MJPG codec
                
            self.is_running = True
            self.start_button.config(text="Kamerayi Durdur", bg='#e74c3c', activebackground='#c0392b')
            self.status_label.config(text="Kamera calisiyor (GPU)", foreground="blue")
            
            # Video thread'ini başlat
            self.video_thread = threading.Thread(target=self.update_video, daemon=True)
            self.video_thread.start()
            
            print("Kamera baslatildi")
            
        except Exception as e:
            error_msg = f"Kamera baslatma hatasi: {str(e)}"
            self.status_label.config(text="Kamera hatasi", foreground="red")
            messagebox.showerror("Hata", error_msg)
    
    def stop_camera(self):
        """Kamerayı durdur"""
        self.is_running = False
        if self.camera:
            self.camera.release()
            self.camera = None
            
        self.start_button.config(text="Kamerayi Baslat", bg='#27ae60', activebackground='#229954')
        self.status_label.config(text="Kamera durduruldu", foreground="green")
        
        # Video frame'i temizle
        self.video_frame.config(
            image="", 
            text="Kamera goruntusu burada gorunecek\n\nYuz tanima icin kamerayi baslatin"
        )
        
        print("Kamera durduruldu")
    
    def update_video(self):
        """Video akışını güncelle - FPS sayacı ile"""
        while self.is_running and self.camera:
            try:
                ret, frame = self.camera.read()
                if not ret:
                    break
                
                # FPS hesapla
                self.fps_counter += 1
                current_time = time.time()
                if current_time - self.fps_start_time >= 1.0:
                    self.current_fps = self.fps_counter
                    self.fps_counter = 0
                    self.fps_start_time = current_time
                    
                    # FPS'i GUI'de güncelle
                    self.root.after(0, lambda: self.fps_label.config(text=f"FPS: {self.current_fps}"))
                    
                # Frame'i işle
                processed_frame = self.process_frame(frame)
                
                # Tkinter için resmi dönüştür
                rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_frame)
                pil_image = pil_image.resize((640, 480), Image.Resampling.LANCZOS)
                
                # PhotoImage oluştur
                photo = ImageTk.PhotoImage(pil_image)
                
                # GUI'yi güncelle (main thread'de)
                self.root.after(0, lambda: self.update_video_frame(photo))
                
            except Exception as e:
                print(f"Video guncelleme hatasi: {str(e)}")
                break
    
    def process_frame(self, frame):
        """Frame'i işle ve yüz tanıma yap - GPU optimizasyonlu"""
        if not self.face_app:
            return frame
            
        try:
            # Frame atlama - performans için
            self.frame_counter += 1
            if self.frame_counter % self.frame_skip != 0:
                # Önceki yüzleri kullan
                return self.draw_previous_faces(frame)
            
            # Frame'i küçült - daha hızlı işlem
            height, width = frame.shape[:2]
            scale = 0.5
            small_frame = cv2.resize(frame, (int(width * scale), int(height * scale)))
            
            # Yüzleri tespit et
            faces = self.face_app.get(small_frame)
            
            # Koordinatları orijinal boyuta geri çevir
            for face in faces:
                face.bbox = face.bbox / scale
            
            self.last_faces = []
            
            for face in faces:
                # Bounding box çiz
                bbox = face.bbox.astype(int)
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                
                # Yüz tanıma
                name, similarity = self.recognize_face(face.embedding)
                
                # Sonuçları sakla
                face_info = {
                    'bbox': bbox,
                    'name': name,
                    'similarity': similarity
                }
                self.last_faces.append(face_info)
                
                # İsim ve benzerlik oranını yaz
                label = f"{name} ({similarity:.2f})" if name != "Unknown" else "Unknown"
                
                # Yazı rengini tanınma durumuna göre ayarla
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                
                # Label arka planı
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                cv2.rectangle(frame, (bbox[0], bbox[1]-label_size[1]-10), 
                            (bbox[0]+label_size[0], bbox[1]), color, -1)
                
                cv2.putText(frame, label, (bbox[0], bbox[1]-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Güven skoru çubuğu
                if name != "Unknown":
                    bar_width = int(100 * similarity)
                    cv2.rectangle(frame, (bbox[0], bbox[3]+5), 
                                (bbox[0] + bar_width, bbox[3]+15), (0, 255, 0), -1)
                
        except Exception as e:
            print(f"Frame isleme hatasi: {str(e)}")
            
        return frame
    
    def draw_previous_faces(self, frame):
        """Önceki frame'deki yüz bilgilerini çiz"""
        for face_info in self.last_faces:
            bbox = face_info['bbox']
            name = face_info['name']
            similarity = face_info['similarity']
            
            # Bounding box çiz
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            
            # İsim yaz
            label = f"{name} ({similarity:.2f})" if name != "Unknown" else "Unknown"
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            
            # Label arka planı
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(frame, (bbox[0], bbox[1]-label_size[1]-10), 
                        (bbox[0]+label_size[0], bbox[1]), color, -1)
            
            cv2.putText(frame, label, (bbox[0], bbox[1]-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Güven skoru çubuğu
            if name != "Unknown":
                bar_width = int(100 * similarity)
                cv2.rectangle(frame, (bbox[0], bbox[3]+5), 
                            (bbox[0] + bar_width, bbox[3]+15), (0, 255, 0), -1)
        
        return frame
    
    def update_video_frame(self, photo):
        """Video frame'ini GUI'de güncelle"""
        if self.is_running:
            self.video_frame.config(image=photo, text="")
            self.video_frame.image = photo  # Referansı koru
    
    def __del__(self):
        """Temizlik işlemleri"""
        if self.camera:
            self.camera.release()
        cv2.destroyAllWindows()