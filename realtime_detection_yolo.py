import os
import cv2
import torch

this_folder = os.path.dirname(os.path.abspath(__file__))
model_name = "best.pt"
model_path = os.path.join(this_folder, model_name)
print(f"Tam Dosya Yolu: {model_path}")

# Kamerayı başlat
cap = cv2.VideoCapture(0)  

# Modeli yükleme fonksiyonu (YOLOv5 ile uyumlu)
def load_model(model_path):
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
    model.eval()  
    return model

# Modeli yükle
model = load_model(model_path)

# Kamera çalışmıyorsa çıkış yap
if not cap.isOpened():
    print("Kamera açılmadı!")
    exit()

# Gerçek zamanlı veri ön işleme
def preprocess_frame(frame):
    # Görüntüyü yeniden boyutlandır
    resized_frame = cv2.resize(frame, (416, 416))  # YOLOv5 varsayılan boyutu
    # RGB formatına dönüştürür
    rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    # Normalizasyon (0-1 aralığına)
    normalized_frame = rgb_frame / 255.0
    return normalized_frame

while True:
    ret, frame = cap.read()
    if not ret:
        print("There is a problem on camera.")
        break

    # Çerçeveyi al
    processed_frame = preprocess_frame(frame)

    # YOLOv5 ile algılama 
    results = model(frame)  # Model çerçeveyi işler
    detections = results.pandas().xyxy[0]

    # Güven oranı eşiği
    confidence_threshold = 0.5

    for index, row in detections.iterrows():
        confidence = row['confidence']
        if confidence < confidence_threshold:
            continue  # Güven oranı düşükse atla

        label = row['name']  # Modelin tespit ettiği sınıf adı
        

        # Koordinatları al
        xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        # Görüntü üzerine kutu çiz
        if label == "with_mask":  # Eğer "with_mask" ise
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(frame, f"{'With Mask'} {confidence:.2f}", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        elif label == "without_mask":  # Eğer "without_mask" ise
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
            cv2.putText(frame, f"{'Without Mask'} {confidence:.2f}", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        

    # Sonuçları göster
    cv2.imshow("Mask Detection", frame)

    # 'q' tuşuna basıldığında çıkış yap
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()