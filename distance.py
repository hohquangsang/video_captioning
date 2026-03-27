# # distance_utils.py
# import torch
# import numpy as np
# import torch.nn.functional as F

# class DistanceCalculator:
#     def __init__(self, midas_model, transform, device):
#         self.midas = midas_model
#         self.transform = transform
#         self.device = device
        
#         # --- CẤU HÌNH ---
#         # Hệ số hiệu chỉnh (Bạn chỉnh số này sau khi đo thực tế)
#         self.CALIBRATION_FACTOR = 250.0

#     def estimate_distance(self, cv_image):
#         """
#         Input: Ảnh OpenCV (numpy array)
#         Output: (Khoảng cách mét, giá trị depth raw)
#         """
#         # 1. Tiền xử lý
#         input_batch = self.transform(cv_image).to(self.device)
        
#         # 2. Inference (Không tính gradient)
#         with torch.no_grad():
#             prediction = self.midas(input_batch)
            
#             # Resize về kích thước gốc
#             prediction = F.interpolate(
#                 prediction.unsqueeze(1),
#                 size=cv_image.shape[:2],
#                 mode="bicubic",
#                 align_corners=False,
#             ).squeeze()
        
#         depth_map = prediction.cpu().numpy()
        
#         # 3. Lấy vùng trung tâm
#         h, w = depth_map.shape
#         center_size = 60
#         y1, y2 = h//2 - center_size, h//2 + center_size
#         x1, x2 = w//2 - center_size, w//2 + center_size
        
#         # Cắt vùng ảnh (nếu ảnh nhỏ hơn vùng cắt thì lấy toàn bộ)
#         if y1 < 0: y1 = 0
#         if x1 < 0: x1 = 0
#         center_roi = depth_map[y1:y2, x1:x2]
        
#         # 4. Tính trung vị
#         if center_roi.size == 0:
#             return 0.0, 0.0
            
#         depth_value = np.median(center_roi)
        
#         # 5. Quy đổi ra mét
#         if depth_value <= 0: return 99.9, depth_value
        
#         distance_meters = self.CALIBRATION_FACTOR / depth_value
        
#         return distance_meters, depth_value

# distance.py
import torch
import numpy as np
import torch.nn.functional as F

class DistanceCalculator:
    def __init__(self, midas_model, transform, device):
        self.midas = midas_model
        self.transform = transform
        self.device = device
        
        # --- CẤU HÌNH HIỆU CHỈNH (CALIBRATION) ---
        # Hệ số này dành cho model DPT_Hybrid.
        # Nếu thấy sai số, bạn chỉnh lại theo công thức:
        # Số_Mới = 800.0 * (1.0 / Số_Mét_Máy_Báo_Tại_1m_Thực_Tế)
        self.CALIBRATION_FACTOR = 800.0 

    def estimate_distance(self, cv_image):
        # 1. Tiền xử lý
        input_batch = self.transform(cv_image).to(self.device)
        
        # 2. Chạy model
        with torch.no_grad():
            prediction = self.midas(input_batch)
            prediction = F.interpolate(
                prediction.unsqueeze(1),
                size=cv_image.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        
        depth_map = prediction.cpu().numpy()
        
        # 3. Lấy vùng trung tâm thông minh (20% ảnh giữa)
        h, w = depth_map.shape
        crop_ratio = 0.2
        
        center_h, center_w = int(h * crop_ratio), int(w * crop_ratio)
        y1, y2 = (h - center_h) // 2, (h - center_h) // 2 + center_h
        x1, x2 = (w - center_w) // 2, (w - center_w) // 2 + center_w
        
        center_roi = depth_map[y1:y2, x1:x2]
        
        # 4. Tính trung vị (Median) để lọc nhiễu
        if center_roi.size == 0: return 0.0, 0.0
        depth_value = np.median(center_roi)
        
        # 5. Quy đổi ra mét
        if depth_value < 0.1: return 99.9, depth_value # Quá xa/Lỗi
        
        distance_meters = self.CALIBRATION_FACTOR / depth_value
        return distance_meters, depth_value