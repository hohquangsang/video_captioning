import os
import json
import random
from PIL import Image
from torch.utils.data import Dataset
import torch.nn as nn

class JsonCaptionsDataset(Dataset):
    def __init__(self, root, annFile, image_transform=None, caption_tokenizer=None,
                 max_len=64, img_key="file_name", cap_key="captions"):
        self.root = root
        self.image_transform = image_transform
        self.caption_tokenizer = caption_tokenizer
        self.max_len = max_len
        self.img_key = img_key
        self.cap_key = cap_key

        with open(annFile, "r", encoding="utf-8") as f:
            data = json.load(f)

        records = data["images"] if isinstance(data, dict) and "images" in data else data

        self.items = []
        for d in records:
            fn = d[self.img_key]
            caps = d.get(self.cap_key, [])
            if not caps:
                continue

            # Đảm bảo tất cả captions là chuỗi
            caps = [str(c).strip() for c in caps if isinstance(c, (str, list))]
            self.items.append((os.path.join(root, fn), caps))

        if not self.items:
            raise RuntimeError("Không tìm thấy mẫu nào. Kiểm tra khóa 'file_name'/'captions' trong file JSON.")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, caps = self.items[idx]
        img = Image.open(img_path).convert("RGB")
        if self.image_transform:
            img = self.image_transform(img)

        # Chọn 1 caption
        caption_text = random.choice(caps)

        if self.caption_tokenizer:
            tokenized_caption = self.caption_tokenizer(
                caption_text,
                padding="max_length",
                truncation=True,
                max_length=self.max_len,
                return_tensors="pt"
            )
            return {
                "image": img,
                "input_ids": tokenized_caption["input_ids"].squeeze(),
                "attention_mask": tokenized_caption["attention_mask"].squeeze()
            }
        else:
            # Fallback for when tokenizer is not provided (e.g., for simple image-caption pairs)
            return {"image": img, "caption": caption_text}


class SampleCaption(nn.Module):
    """Chọn ngẫu nhiên 1 caption (và ép về string an toàn)."""
    def __call__(self, sample):
        if isinstance(sample, list) and len(sample) > 0:
            cap = random.choice(sample)
            if isinstance(cap, list):  # Trường hợp lồng
                cap = cap[0]
            return str(cap)
        return str(sample)
