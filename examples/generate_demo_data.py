from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


POSITIVE_TEXTS = [
    "数字导览讲解清晰，文化故事更容易理解，沉浸感很强。",
    "AI讲解员互动自然，游客停留时间明显变长，体验很好。",
    "展陈内容和智能推荐结合得不错，既方便又有文化深度。",
    "沉浸式光影与智能问答配合顺畅，参观过程很投入。",
]

NEGATIVE_TEXTS = [
    "导览响应慢，推荐内容重复，影响游览节奏。",
    "交互功能不稳定，游客体验一般，文化信息传达不够充分。",
    "设备识别偶尔出错，沉浸感不足，停留意愿较低。",
    "智能服务形式大于内容，文化理解帮助有限。",
]


def create_demo_image(path: Path, score: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (256, 256), (245, 243, 236))
    draw = ImageDraw.Draw(image)
    radius = int(30 + score * 70)
    color = (60, 130 + int(score * 80), 180 - int(score * 40))
    draw.rectangle((30, 180, 226, 220), fill=(120, 90, 50))
    draw.ellipse((128 - radius, 90 - radius, 128 + radius, 90 + radius), fill=color)
    image.save(path)


def build_demo_dataset(output_csv: str = "examples/demo_dataset.csv", size: int = 40) -> None:
    random.seed(42)
    np.random.seed(42)
    rows = []
    image_dir = Path("examples/demo_images")
    for idx in range(size):
        cultural_score = np.clip(np.random.normal(0.7, 0.15), 0.1, 1.0)
        engagement_score = np.clip(np.random.normal(0.65, 0.18), 0.1, 1.0)
        service_efficiency = np.clip(np.random.normal(0.68, 0.12), 0.1, 1.0)
        social_value = np.clip(np.random.normal(0.62, 0.2), 0.1, 1.0)
        text = random.choice(POSITIVE_TEXTS if cultural_score + engagement_score > 1.3 else NEGATIVE_TEXTS)
        target_score = (
            0.28 * cultural_score
            + 0.27 * engagement_score
            + 0.25 * service_efficiency
            + 0.20 * social_value
            + np.random.normal(0, 0.03)
        )
        image_path = image_dir / f"scene_{idx:03d}.png"
        create_demo_image(image_path, float(target_score))
        rows.append(
            {
                "review_text": text,
                "image_path": str(image_path.resolve()),
                "tech_empowerment": round(service_efficiency * 100, 2),
                "visitor_experience": round(engagement_score * 100, 2),
                "cultural_value": round(cultural_score * 100, 2),
                "economic_social_gain": round(social_value * 100, 2),
                "interaction_count": int(10 + engagement_score * 25 + np.random.randint(0, 5)),
                "stay_duration": round(15 + cultural_score * 45 + np.random.normal(0, 3), 2),
                "target_score": round(float(target_score * 100), 2),
            }
        )
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_csv, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    build_demo_dataset()
