import numpy as np


class Element:
    # Class containing information about a specific crop
    def __init__(
        self, image: np.ndarray, image_id: int, file_name: str, width: float, height: float
    ) -> None:
        self.image_id = image_id
        self.file_name = file_name
        self.width = width
        self.height = height
        self.image = image  # Original image
        self.category_names = []  # CVAT label name for each annotation
        self.bbox = []  # List of [x, y, width, height]
        self.detected_masks = []  # List of polygons (flat [x1, y1, x2, y2, ...])
        self.areas = []  # List of areas of bbox/masks depending on the task
        self.iscrowd = 0  # 0 | 1 object is not a group | group
