import cv2
import numpy as np
import torch
from tqdm import tqdm


DEFAULT_SEMANTIC_MODEL = "nvidia/segformer-b2-finetuned-cityscapes-1024-1024"


def normalize_class_name(name):
    return str(name).strip().lower().replace("_", " ")


class SemanticSegmenter:
    """
    Generates CVAT polygon annotations with a semantic segmentation model.

    The default model is SegFormer-B2 fine-tuned on Cityscapes at 1024x1024,
    whose label set contains road, sidewalk, building, vegetation and sky.
    Every connected region of a mapped class becomes one polygon annotation.

    Args:
        elements (list): Dataset created by DataGen.
        class_map (dict): Model class name -> CVAT label name.
        model_path (str): Hugging Face model id or local path.
        conf (float): Default per-pixel probability threshold.
        conf_dict (dict): Per-CVAT-label probability thresholds.
        minimize_points (bool): Simplify polygon contours.
        min_area (float): Ignore regions smaller than this many pixels.
        img_size (int, optional): Square input size for the segmentation model.
    """

    def __init__(
        self,
        elements,
        class_map,
        model_path=None,
        conf=0.4,
        conf_dict=None,
        minimize_points=True,
        min_area=200,
        img_size=None,
        exclude_masks=None,
        device=None,
    ) -> None:
        from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

        self.elements = elements
        self.conf = conf
        self.conf_dict = conf_dict or {}
        self.minimize_points = minimize_points
        self.min_area = min_area
        self.img_size = img_size
        # image_id -> binary mask of instance regions that stuff must not cover
        self.exclude_masks = exclude_masks or {}
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        model_path = model_path or DEFAULT_SEMANTIC_MODEL
        print(f"Loading semantic segmentation model '{model_path}' on {self.device}...")
        self.processor = AutoImageProcessor.from_pretrained(model_path)
        self.model = AutoModelForSemanticSegmentation.from_pretrained(model_path)
        self.model.to(self.device).eval()

        self.label_to_index = {}
        for index, label in self.model.config.id2label.items():
            self.label_to_index.setdefault(normalize_class_name(label), int(index))

        # CVAT label name -> list of model class indices
        self.label_to_indices = self._resolve_class_map(class_map)

    def _resolve_class_map(self, class_map):
        if not class_map or any(isinstance(key, int) for key in class_map):
            raise ValueError(
                "The semantic backend requires a 'mapping' section with model class "
                "names (e.g. 'vegetation: vegetation'), not class indices."
            )

        label_to_indices = {}
        unknown = []
        for model_name, label in class_map.items():
            index = self.label_to_index.get(normalize_class_name(model_name))
            if index is None:
                unknown.append(model_name)
            else:
                label_to_indices.setdefault(str(label), set()).add(index)

        if unknown:
            available = ", ".join(
                sorted(self.model.config.id2label.values())[:40]
            )
            raise ValueError(
                f"Classes {unknown} are not present in model "
                f"'{self.model.config.name_or_path}'. Available classes (first 40): "
                f"{available}"
            )
        return {label: sorted(indices) for label, indices in label_to_indices.items()}

    def _conf_for(self, label):
        return float(self.conf_dict.get(label, self.conf))

    def process(self):
        for element in tqdm(self.elements, desc="Processing images", unit="image", total=len(self.elements)):
            height, width = element.image.shape[:2]
            rgb = cv2.cvtColor(element.image, cv2.COLOR_BGR2RGB)

            processor_kwargs = {"return_tensors": "pt"}
            if self.img_size:
                processor_kwargs["size"] = {
                    "height": int(self.img_size),
                    "width": int(self.img_size),
                }
            inputs = self.processor(images=rgb, **processor_kwargs)
            inputs = {
                key: value.to(self.device) if torch.is_tensor(value) else value
                for key, value in inputs.items()
            }

            with torch.no_grad():
                outputs = self.model(**inputs)

            # Softmax at the model resolution, then upsample the probability of
            # every mapped label to the original image size. Argmax over the
            # upsampled probabilities gives smooth, non-overlapping regions.
            probs = outputs.logits.softmax(dim=1)[0].cpu().numpy()
            labels = list(self.label_to_indices)
            probability_maps = [
                cv2.resize(
                    probs[self.label_to_indices[label]].max(axis=0),
                    (width, height),
                    interpolation=cv2.INTER_LINEAR,
                )
                for label in labels
            ]
            stacked = np.stack(probability_maps)
            best_label = stacked.argmax(axis=0)
            best_probability = stacked.max(axis=0)

            bboxes = []
            masks = []
            areas = []
            category_names = []

            exclusion = self.exclude_masks.get(element.image_id)

            for label_index, label in enumerate(labels):
                class_mask = (
                    (best_label == label_index)
                    & (best_probability >= self._conf_for(label))
                ).astype(np.uint8)
                if exclusion is not None:
                    class_mask[exclusion > 0] = 0
                if class_mask.max() == 0:
                    continue

                contours, _ = cv2.findContours(
                    class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )

                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area < self.min_area:
                        continue

                    points = contour
                    if self.minimize_points:
                        epsilon = 0.002 * cv2.arcLength(contour, True)
                        points = cv2.approxPolyDP(contour, epsilon, True)

                    flat_points = points.reshape(-1).astype(float).tolist()
                    if len(flat_points) < 6:
                        continue

                    x, y, w, h = cv2.boundingRect(points)
                    bboxes.append([float(x), float(y), float(w), float(h)])
                    masks.append(flat_points)
                    areas.append(float(area))
                    category_names.append(label)

            element.bbox = bboxes
            element.detected_masks = masks
            element.areas = areas
            element.category_names = category_names
            element.iscrowd = 0
        return self.elements
