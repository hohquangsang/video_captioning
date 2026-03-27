#dataset path
data_root = "/content/drive/MyDrive/Image_captioning/flickr8k"
image_dir = f"{data_root}/Images"
caption_dir = f"{data_root}/captions/caption_8k.json"
#train set and val set
train_image_path = image_dir
train_caption_path = caption_dir
val_image_path = image_dir
val_caption_path = caption_dir
#ViT config and transformer config
# Using smaller, diagnostic configuration to troubleshoot CUDA error
# vit_cfg = dict(
#     image_size=224,
#     patch_size=16,
#     in_channels=3,
#     embed_dim=128, # Reduced dimension
#     depth=2,       # Reduced depth
#     num_heads=4,   # Reduced heads
#     mlp_ratio=2.0,
#     dropout=0.1
# )

# trans_cfg = dict(
#     dim=128,       # Reduced dimension
#     num_heads=4,   # Reduced heads
#     num_layers=2,  # Reduced layers
#     ff_dim=256,    # Reduced feed-forward dimension
#     dropout=0.1,
#     max_len=10     # Reduced max_len
# )
vit_cfg = dict(
    image_size=224,      # Kích thước ảnh đầu vào (Cố định)
    patch_size=16,       # Chia ảnh thành các ô 16x16 (Chi tiết tốt hơn 32)
    in_channels=3,       # Ảnh màu RGB
    embed_dim=512,       # Kích thước vector đặc trưng (Medium size)
    depth=6,             # Số lớp Encoder (6 lớp là đủ cho 5GB dữ liệu)
    num_heads=8,         # Số lượng đầu Attention (512 / 8 = 64 dim/head -> OK)
    mlp_ratio=4.0,       # Hệ số mở rộng trong lớp FeedForward
    dropout=0.1          # Giảm Overfitting
)

# --- Cấu hình Decoder (Transformer) ---
trans_cfg = dict(
    dim=512,             # Kích thước vector đầu vào (Khớp với ViT để ko cần Project)
    num_heads=8,         # Khớp với ViT
    num_layers=6,        # Số lớp Decoder (Cân bằng với Encoder)
    ff_dim=2048,         # Thường gấp 4 lần dim (512 * 4)
    dropout=0.1,         
    max_len=40           # Độ dài tối đa của câu caption (Flickr30k câu khá dài)
)
epochs = 20