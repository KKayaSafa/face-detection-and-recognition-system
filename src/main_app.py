import tkinter as tk
from tkinter import ttk, messagebox

# Modülleri import et
from person_counter import PersonCounterApp
from face_recognition_app import FaceRecognitionApp

class MainApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Bilgisayar Görü Uygulamalari - Ana Menü")
        self.root.geometry("800x600")
        self.root.configure(bg='#f0f0f0')
        
        # Aktif uygulama penceresi
        self.active_window = None
        
        # Ana GUI'yi oluştur
        self.create_main_gui()
        
    def create_main_gui(self):
        """Ana menü arayüzünü oluştur"""
        # Ana frame
        main_frame = ttk.Frame(self.root, padding="30")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Başlık
        title_label = tk.Label(
            main_frame, 
            text="Bilgisayar Goru Uygulamalari",
            font=("Arial", 24, "bold"),
            fg='#2c3e50'
        )
        title_label.pack(pady=(0, 30))
        
        
        # Butonlar için frame
        buttons_frame = ttk.Frame(main_frame)
        buttons_frame.pack(pady=20)
        
        # Kişi sayma butonu
        person_count_btn = tk.Button(
            buttons_frame,
            text="Kisi Sayma\n(YOLOv8n)",
            font=("Arial", 14, "bold"),
            bg='#3498db',
            fg='white',
            activebackground='#2980b9',
            activeforeground='white',
            relief='raised',
            bd=3,
            padx=30,
            pady=20,
            width=25,
            height=4,
            command=self.open_person_counter
        )
        person_count_btn.pack(pady=15)
        
        # Yüz tanıma butonu
        face_recognition_btn = tk.Button(
            buttons_frame,
            text="Yuz Tanima\n(InsightFace)",
            font=("Arial", 14, "bold"),
            bg='#e74c3c',
            fg='white',
            activebackground='#c0392b',
            activeforeground='white',
            relief='raised',
            bd=3,
            padx=30,
            pady=20,
            width=25,
            height=4,
            command=self.open_face_recognition
        )
        face_recognition_btn.pack(pady=15)
        
        # Çıkış butonu
        exit_btn = tk.Button(
            buttons_frame,
            text="Cikis",
            font=("Arial", 12),
            bg="#645f5e",
            fg='white',
            activebackground="#e64818",
            activeforeground='white',
            relief='raised',
            bd=2,
            padx=20,
            pady=10,
            width=25,
            command=self.exit_app
        )
        exit_btn.pack(pady=(30, 0))
        
        # Bilgi etiketi
        info_label = tk.Label(
            main_frame,
            text="Uygulamalar kamera erisimi gerektirir\nGPU destegi icin CUDA kurulu olmalidir",
            font=("Arial", 10),
            bg='#f0f0f0',
            fg='#7f8c8d',
            justify='center'
        )
        info_label.pack(side=tk.BOTTOM, pady=(40, 0))
        
    def open_person_counter(self):
        """Kişi sayma uygulamasını aç"""
        try:
            if self.active_window:
                self.active_window.destroy()
                
            # Yeni pencere oluştur
            counter_window = tk.Toplevel(self.root)
            counter_window.title("Kisi Sayma Uygulamasi")
            counter_window.geometry("1200x900")
            
            # Pencere kapatılırken temizlik yap
            def on_closing():
                counter_window.destroy()
                self.active_window = None
                
            counter_window.protocol("WM_DELETE_WINDOW", on_closing)
            
            # Kişi sayma uygulamasını başlat
            self.active_window = counter_window
            person_app = PersonCounterApp(counter_window)
            
        except Exception as e:
            messagebox.showerror("Hata", f"Kisi sayma uygulamasi acilirken hata oluştu:\n{str(e)}")
            
    def open_face_recognition(self):
        """Yüz tanıma uygulamasını aç"""
        try:
            if self.active_window:
                self.active_window.destroy()
                
            # Yeni pencere oluştur
            face_window = tk.Toplevel(self.root)
            face_window.title("Yüz Tanima Uygulamasi")
            face_window.geometry("1200x900")
            
            # Pencere kapatılırken temizlik yap
            def on_closing():
                face_window.destroy()
                self.active_window = None
                
            face_window.protocol("WM_DELETE_WINDOW", on_closing)
            
            # Yüz tanıma uygulamasını başlat
            self.active_window = face_window
            face_app = FaceRecognitionApp(face_window)
            
        except Exception as e:
            messagebox.showerror("Hata", f"Yüz tanıma uygulaması açılırken hata oluştu:\n{str(e)}")
            
    def exit_app(self):
        """Uygulamadan çık"""
        if messagebox.askyesno("Cikis", "Uygulamayi kapatmak istediginizden emin misiniz?"):
            if self.active_window:
                self.active_window.destroy()
            self.root.quit()
            self.root.destroy()

#GEREKSİZ CİBİİİİİİİİİİİİİİİİİİİİİİİİİİİİİİİİİİİİİİİİİİİİİİİİİİİİİİİİİİİİii
def check_dependencies():
    """Gerekli kütüphanelerin kontrolü"""
    missing_packages = []
    
    # Temel kütüphaneler
    try:
        import cv2
    except ImportError:
        missing_packages.append("opencv-python")
        
    try:
        from ultralytics import YOLO
    except ImportError:
        missing_packages.append("ultralytics")
        
    try:
        from insightface import app as face_app
    except ImportError:
        missing_packages.append("insightface")
        
    try:
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError:
        missing_packages.append("scikit-learn")
        
    try:
        import numpy as np
    except ImportError:
        missing_packages.append("numpy")
        
    try:
        from PIL import Image
    except ImportError:
        missing_packages.append("Pillow")
    
    if missing_packages:
        error_msg = f"""
Eksik Python paketleri tespit edildi:
{', '.join(missing_packages)}

Lutfen su komutu calistirin:
pip install {' '.join(missing_packages)}
        """
        print(error_msg)
        return False
        
    return True

def main():
    """Ana uygulama başlatıcı"""
    print("Bilgisayar Goru Uygulamalari baslatiliyor...")
    print("=" * 50)
    
    # Bağımlılık kontrolü
    if not check_dependencies():
        print("Gerekli paketler eksik!")
        input("Devam etmek icin Enter'a basin...")
        return
        
    print("Tum bagimliliklar mevcut")
    print("Ana menu aciliyor...")
    
    # Ana pencereyi oluştur
    root = tk.Tk()
    
    try:
        app = MainApp(root)
        root.mainloop()
    except KeyboardInterrupt:
        print("\nKullanici tarafindan durduruldu")
    except Exception as e:
        print(f"Beklenmeyen hata: {str(e)}")
    finally:
        print("Uygulama sonlandirildi")
        try:
            root.destroy()
        except:
            pass

if __name__ == "__main__":
    main()