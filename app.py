from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import torch
import numpy as np
from PIL import Image
from transformers import T5Tokenizer
from torchvision import transforms
# --- THAY ĐỔI: Dùng deep_translator thay vì googletrans ---
from deep_translator import GoogleTranslator
import io
import base64

# --- IMPORT CÁC MODULE CỦA DỰ ÁN ---
try:
    from config import vit_cfg, trans_cfg
    from src.main.model.model import ViT_Transformer
    from distance import DistanceCalculator
except ImportError as e:
    print("LỖI IMPORT: Vui lòng kiểm tra cấu trúc thư mục.")
    print(f"Chi tiết: {e}")
    exit(1)

app = Flask(__name__)
CORS(app)

# Tự động chọn thiết bị
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"--- Đang chạy trên thiết bị: {device.upper()} ---")

# ==========================================
# 1. LOAD MODEL CAPTIONING (CUSTOM VIT-T5)
# ==========================================
print(">> Đang tải Tokenizer (T5-base)...")
try:
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    if tokenizer.bos_token_id is None:
        tokenizer.bos_token_id = tokenizer.pad_token_id
except Exception as e:
    print(f"Lỗi tải Tokenizer: {e}")
    exit(1)

print(">> Đang khởi tạo Model ViT_Transformer...")
model_custom = ViT_Transformer(vit_cfg, trans_cfg, vocab_size=len(tokenizer)).to(device)

CHECKPOINT_PATH = "model_epoch_50.pth" 

print(f">> Đang nạp trọng số từ {CHECKPOINT_PATH}...")
try:
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model_custom.load_state_dict(checkpoint['model_state_dict'])
    else:
        model_custom.load_state_dict(checkpoint)
    
    model_custom.eval()
    print(">> Load Custom Model thành công!")
except FileNotFoundError:
    print(f"LỖI: Không tìm thấy file '{CHECKPOINT_PATH}'.")
    exit(1)
except Exception as e:
    print(f"LỖI LOAD MODEL: {e}")
    exit(1)

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ==========================================
# 2. LOAD MODEL KHOẢNG CÁCH (DPT_HYBRID)
# ==========================================
print(">> Đang tải MiDaS (DPT_Hybrid)...")
try:
    midas_model = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform
except Exception as e:
    print(f"Không tải được DPT_Hybrid, chuyển sang MiDaS_small. Lỗi: {e}")
    midas_model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform

midas_model.to(device)
midas_model.eval()

dist_calc = DistanceCalculator(midas_model, midas_transforms, device)

# --- THAY ĐỔI: Khởi tạo dịch thuật mới ---
# Không cần khởi tạo object Translator() như cũ nữa

# ==========================================
# 3. CÁC ROUTE XỬ LÝ (API)
# ==========================================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Nhận dữ liệu ảnh
        data = request.json
        if 'image' not in data:
            return jsonify({'error': 'Không tìm thấy ảnh gửi lên'}), 400
            
        image_data = data['image'].split(",")[1]
        image_bytes = base64.b64decode(image_data)
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        cv_image = np.array(pil_image)

        # 2. Đo khoảng cách
        distance_meters, _ = dist_calc.estimate_distance(cv_image)
        
        dist_text = f"{distance_meters:.1f}m"
        warning_msg = ""
        prefix_speech = "Phía trước là "
        
        if distance_meters < 0.8:
            warning_msg = "CẨN THẬN! RẤT GẦN"
            prefix_speech = "Nguy hiểm! Ngay trước mặt là "
        elif distance_meters < 1.5:
            prefix_speech = "Khá gần, một "

        # 3. Sinh Caption
        img_tensor = image_transform(pil_image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            caption_en = model_custom.generate(
                img_tensor, 
                tokenizer, 
                max_len=30,
                device=device,
                temperature=1.0,
                top_k=5,
                top_p=0.9
            )
        
        # 4. Dịch sang tiếng Việt (SỬ DỤNG deep_translator)
        try:
            # Code mới: Ổn định hơn googletrans
            caption_vi = GoogleTranslator(source='en', target='vi').translate(caption_en)
        except Exception as e:
            print(f"LỖI DỊCH THUẬT: {e}") # In lỗi ra để biết tại sao
            caption_vi = caption_en # Fallback về tiếng Anh

        # 5. Tạo câu nói
        distance_cm = int(distance_meters * 100)
        final_speech = f"{prefix_speech} {caption_vi} cách {distance_cm} xăng ti mét"

        return jsonify({
            'caption_vi': caption_vi,
            'distance': dist_text,
            'warning': warning_msg,
            'final_speech': final_speech
        })

    except Exception as e:
        print(f"LỖI SERVER: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)


# # ==========================================
# # 1. LOAD MODEL CỦA BẠN (CUSTOM MODEL)
# # ==========================================
# print(">> Đang tải Tokenizer (T5-small)...")
# tokenizer = T5Tokenizer.from_pretrained("t5-small")
# if tokenizer.bos_token_id is None:
#     tokenizer.bos_token_id = tokenizer.pad_token_id

# print(">> Đang khởi tạo kiến trúc ViT_Transformer...")
# # Khởi tạo model với cấu hình y hệt lúc train
# # vocab_size lấy từ len(tokenizer)
# model_custom = ViT_Transformer(vit_cfg, trans_cfg, len(tokenizer)).to(device)

# print(f">> Đang nạp trọng số từ model_epoch_390.pth...")
# try:
#     # Load state_dict (map_location để tránh lỗi nếu train GPU mà chạy CPU)
#     checkpoint = torch.load("model_epoch_390.pth", map_location=device)
    
#     # Đôi khi file lưu cả optimizer, epoch... nên cần check xem nó là dict trơn hay nested
#     if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
#         model_custom.load_state_dict(checkpoint['model_state_dict'])
#     else:
#         model_custom.load_state_dict(checkpoint)
        
#     model_custom.eval() # Chuyển sang chế độ dự đoán (tắt dropout)
#     print(">> Load model thành công!")
# except Exception as e:
#     print(f"LỖI LOAD MODEL: {e}")
#     print("Vui lòng kiểm tra lại path file .pth hoặc config.py")

# # Tạo bộ tiền xử lý ảnh (Transform) giống file model.py
# # Resize về 224x224 là bắt buộc với ViT
# image_transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
# ])

# # ==========================================
# # 2. LOAD MIDAS (ĐO KHOẢNG CÁCH)
# # ==========================================
# print(">> Đang tải MiDaS...")
# midas_model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
# midas_model.to(device)
# midas_model.eval()
# midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform

# dist_calc = DistanceCalculator(midas_model, midas_transforms, device)
# translator = Translator()

# # ==========================================
# # 2. LOAD MIDAS (ĐO KHOẢNG CÁCH) - ĐÃ NÂNG CẤP
# # ==========================================
# print(">> Đang tải MiDaS (DPT_Hybrid)...")

# # Thay 'MiDaS_small' bằng 'DPT_Hybrid' để chính xác hơn
# # (Nếu máy quá yếu thì đổi lại 'MiDaS_small')
# midas_model = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid") 

# midas_model.to(device)
# midas_model.eval()

# # Load transform tương ứng cho DPT
# midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform

# dist_calc = DistanceCalculator(midas_model, midas_transforms, device)
# translator = Translator()

# # ==========================================
# # 3. ROUTE XỬ LÝ
# # ==========================================

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         # 1. Nhận ảnh
#         data = request.json
#         image_data = data['image'].split(",")[1]
#         image_bytes = base64.b64decode(image_data)
        
#         pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
#         cv_image = np.array(pil_image)

#         # 2. Đo khoảng cách (Logic cũ)
#         distance_meters, _ = dist_calc.estimate_distance(cv_image)
        
#         # Logic cảnh báo
#         dist_text = f"{distance_meters:.1f}m"
#         warning_msg = ""
#         prefix_speech = "Phía trước là" 
#         if distance_meters < 0.8:
#             warning_msg = "CẨN THẬN! RẤT GẦN"
#             prefix_speech = "Nguy hiểm! Ngay trước mặt là "
#         elif distance_meters < 1.5:
#             prefix_speech = "Khá gần, một "

#         # 3. Sinh Caption bằng MODEL CỦA BẠN
#         # Tiền xử lý ảnh
#         img_tensor = image_transform(pil_image).unsqueeze(0).to(device)
        
#         with torch.no_grad():
#             # Gọi hàm generate của model (như trong train.py bạn đã dùng)
#             # Hàm generate này nằm trong src/main/model.py (hoặc ViT class)
#             caption_generated = model_custom.generate(
#                 img_tensor, 
#                 tokenizer, 
#                 max_len=20, # Có thể chỉnh độ dài
#                 device=device
#             #     top_k=1,          # Quan trọng: Chỉ lấy 1 từ tốt nhất
#             #     temperature=1.0,  # Giữ nguyên nhiệt độ
#             #     top_p=0           # Tắt Top-P để dùng Top-K thuần túy
#             )
        
#         # Nếu nó trả về list token, cần decode. Nhưng theo train.py thì nó trả về text.
#         caption_en = caption_generated
        
#         # Dịch sang tiếng Việt
#         try:
#             caption_vi = translator.translate(caption_en, src='en', dest='vi').text
#         except:
#             caption_vi = caption_en

#         final_speech = f"{prefix_speech} {caption_vi} cách {int(distance_meters*100)} xăng ti mét"

#         return jsonify({
#             'caption_vi': caption_vi,
#             'distance': dist_text,
#             'warning': warning_msg,
#             'final_speech': final_speech
#         })

#     except Exception as e:
#         print(f"LỖI: {e}")
#         return jsonify({'error': str(e)}), 500

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000, debug=True)
