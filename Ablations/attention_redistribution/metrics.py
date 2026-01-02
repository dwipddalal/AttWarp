from typing import Tuple, List
import numpy as np

BBOX_METHODS: List[str] = ["xywh", "xyxy"]

def interpret_bbox(b, method: str) -> Tuple[int, int, int, int]:

    if method == "xywh":
        x, y, w, h = b
        return int(x), int(y), int(w), int(h)
    x1, y1, x2, y2 = b
    return int(x1), int(y1), int(x2 - x1), int(y2 - y1)


def top1(attention_map: np.ndarray, box: Tuple[int, int, int, int]) -> int:

    x0, y0, w, h = box
    r, c = np.unravel_index(np.argmax(attention_map), attention_map.shape)
    return 1 if (x0 <= c < x0 + w and y0 <= r < y0 + h) else 0


def am_all(attention_map: np.ndarray, box: Tuple[int, int, int, int]) -> float:

    x0, y0, w, h = box
    return attention_map[y0:y0 + h, x0:x0 + w].sum() / (attention_map.sum() + 1e-12)


