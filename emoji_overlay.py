"""인식된 제스처 이름 -> 컬러 이모지 이미지.

다른 제스처에도 이모지를 붙이려면 GESTURE_EMOJI에 "라벨": "이모지"를 추가하면 된다.
"""
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFont

GESTURE_EMOJI = {
    "rock": "🪨",       # 바위
    "scissors": "✂",    # 가위
    "paper": "✋",       # 보 (손바닥)
}

EMOJI_FONT = "C:/Windows/Fonts/seguiemj.ttf"  # Windows 기본 컬러 이모지 폰트


def emoji_for(label):
    return GESTURE_EMOJI.get(label.lower()) if label else None


@lru_cache(maxsize=64)
def emoji_image(label, size):
    """라벨에 해당하는 이모지를 size x size RGBA 이미지로 반환 (없으면 None)."""
    char = emoji_for(label)
    if char is None:
        return None
    # 이모지 변형 선택자(U+FE0F)는 Pillow 기본 레이아웃에서 폭으로 계산돼 이모지가 밀려 잘리므로 제거
    char = char.replace("\ufe0f", "")
    try:
        font = ImageFont.truetype(EMOJI_FONT, int(size * 0.8))
    except OSError:
        return None
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    ImageDraw.Draw(img).text((size / 2, size / 2), char, font=font,
                             embedded_color=True, anchor="mm")
    return img


def paste_emoji_bgr(frame, label, x, y, size):
    """OpenCV BGR 프레임의 (x, y) 좌상단에 이모지를 알파 합성 (화면 밖은 잘라냄)."""
    img = emoji_image(label, size)
    if img is None:
        return
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + size, w), min(y + size, h)
    if x0 >= x1 or y0 >= y1:
        return
    rgba = np.asarray(img)[y0 - y:y1 - y, x0 - x:x1 - x].astype(np.float32)
    alpha = rgba[:, :, 3:4] / 255.0
    roi = frame[y0:y1, x0:x1].astype(np.float32)
    frame[y0:y1, x0:x1] = (rgba[:, :, 2::-1] * alpha + roi * (1 - alpha)).astype(np.uint8)
