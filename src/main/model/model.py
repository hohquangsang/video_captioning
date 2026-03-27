from torch import nn
from src.main.encode.vit import ViT
from src.main.decode.transformer import Transformer
import torch
import torch.nn.functional as F
2

class ViT_Transformer(nn.Module):
    def __init__(
        self,
        vit_config: dict,
        trans_cfg: dict,
        vocab_size: int,
        max_len: int = 32, # Tăng max_len để linh hoạt hơn
    ):
        super().__init__()

        # 1. Encoder: ViT
        self.encoder = ViT(
            image_size=vit_config.get("image_size", 224),
            patch_size=vit_config.get("patch_size", 32),
            in_channels=vit_config.get("in_channels", 3),
            embed_dim=vit_config.get("embed_dim", 768),
            depth=vit_config.get("depth", 12),
            num_heads=vit_config.get("num_heads", 12),
            mlp_ratio=vit_config.get("mlp_ratio", 4.0),
            dropout=vit_config.get("dropout", 0.1),
            num_classes=1
        )

        # 2. Decoder: Transfomer
        self.decoder = Transformer(
            vocab_size=vocab_size,
            dim=trans_cfg.get("dim", 512),
            num_heads=trans_cfg.get("num_heads", 8),
            num_layers=trans_cfg.get("num_layers", 6),
            ff_dim=trans_cfg.get("ff_dim", 2048),
            dropout=trans_cfg.get("dropout", 0.1),
            max_len=trans_cfg.get("max_len", max_len)
        )

        # 3. Projection Layer
        vit_dim = vit_config.get("embed_dim", 768)
        trans_dim = trans_cfg.get("dim", 512)
        if vit_dim != trans_dim:
            self.proj = nn.Linear(vit_dim, trans_dim)
        else:
            self.proj = nn.Identity()

    def forward(self, images, input_ids, mask=None):
        """
        images: tensor [B, 3, H, W]
        input_ids: tensor [B, T] (chuỗi token caption)
        mask: tensor [B, T] (attention mask cho padding, 1 cho token thật, 0 cho padding)
        """
        # Encoder: trích xuất đặc trưng ảnh
        _, features = self.encoder(images)  # [B, num_patches+1, embed_dim]

        # Chiếu sang không gian decoder
        encoder_out = self.proj(features)

        # Decoder: sinh caption
        T = input_ids.size(1)

        # 1. Tạo Look-ahead mask (tam giác dưới)
        tgt_mask = torch.tril(torch.ones(T, T, device=input_ids.device)).unsqueeze(0).unsqueeze(0)

        # 2. Kết hợp với Padding mask (nếu có)
        if mask is not None:
            # Chuyển mask (B, T) thành (B, 1, 1, T) để nhân (sử dụng broadcast)
            # Mask padding (0) sẽ biến cột/hàng tương ứng thành 0 trong tgt_mask
            tgt_mask = tgt_mask * mask.unsqueeze(1).unsqueeze(1)

        # 3. Chuyển sang dạng Attention Mask (giá trị -inf)
        # T5Block/Attention block của tôi chấp nhận mask có giá trị -inf.
        # Nếu tgt_mask * mask có giá trị 0, nghĩa là vị trí đó không hợp lệ.
        attention_mask = tgt_mask.float().masked_fill(tgt_mask == 0, float('-inf'))

        # Gọi decoder
        logits = self.decoder(input_ids, encoder_out, attention_mask)
        return logits

    @torch.no_grad()
    def generate(self, image, tokenizer, max_len=32, device="cpu", temperature=1.0, top_k=0, top_p=0.9):
        """
        Sinh caption (sử dụng Sampling để khắc phục lỗi lặp lại)
        """
        self.eval()

        # Xử lý Encoder Output
        _, features = self.encoder(image)
        encoder_out = self.proj(features)

        # 1. Khởi tạo bằng BOS token
        start_token_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.pad_token_id
        input_ids = torch.tensor([[start_token_id]], device=device)

        # Giới hạn max_len
        actual_max_len = min(self.decoder.max_len, max_len)

        for _ in range(actual_max_len - 1):
            T = input_ids.size(1)

            # 2. Tạo Mask Tam giác dưới (Look-ahead Mask) trong mỗi bước
            # Shape: [1, 1, T, T]. Đây là mask chính xác cần thiết cho self-attention.
            tgt_mask = torch.tril(torch.ones(T, T, device=device)).unsqueeze(0).unsqueeze(0)

            # Chuyển sang dạng Attention Mask (-inf)
            attention_mask = tgt_mask.float().masked_fill(tgt_mask == 0, float('-inf'))

            # Gọi decoder
            logits = self.decoder(input_ids, encoder_out, attention_mask)

            # Lấy logits của token cuối cùng và áp dụng temperature
            next_token_logits = logits[:, -1, :] / temperature

            # 3. Áp dụng Sampling (Top-k và Top-p)
            # Nếu temperature thấp (ví dụ 0.1) và không dùng sampling, nó sẽ quay lại Greedy Decoding (lặp lại)
            if top_k > 0:
                v, _ = torch.topk(next_token_logits, top_k)
                next_token_logits[next_token_logits < v[:, [-1]]] = -float('Inf')

            if top_p > 0:
                # Áp dụng Nucleus Sampling (Top-p)
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[:, indices_to_remove] = -float('Inf')

            # Sampling từ phân phối đã điều chỉnh
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Nối token mới vào chuỗi
            input_ids = torch.cat([input_ids, next_token], dim=1)

            if next_token.item() == tokenizer.eos_token_id:
                break

        caption = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        return caption