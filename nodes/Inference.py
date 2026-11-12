import numpy as np
from ultralytics import YOLO
from tqdm import tqdm
import cv2


def normalize_class_name(name):
    """Normalize class names so 'traffic_light' and 'Traffic Light' match."""
    return str(name).strip().lower().replace("_", " ")


class Inferencer:
    """
    Class for performing inference using a YOLO model and processing the results.

    Args:
        elements (DataGen): Dataset for inference.
        model_path (str, optional): Path to the YOLO model file. Defaults to "yolov8m.pt".
        imgsz (int, optional): Input image size. Defaults to 640.
        conf (float, optional): Default confidence threshold. Defaults to 0.4.
        iou (float, optional): Intersection over Union (IoU) threshold. Defaults to 0.8.
        model (YOLO, optional): Pre-initialized YOLO model. Defaults to None.
        class_map (dict, optional): Model class index (int) or model class name (str)
            mapped to a CVAT label name. Name-based mapping is recommended because
            model class indices depend on the exact weights file.
        conf_dict (dict, optional): Per-label (or per-model-class-name) confidence.
    """

    def __init__(
        self,
        elements,
        segment=False,
        model_path="yolov8m.pt",
        imgsz=640,
        conf=0.4,
        iou=0.8,
        model=None,
        class_map=None,
        minimize_points=True,
        use_box_propt_sam=False,
        conf_dict={},
    ) -> None:
        self.segment = segment
        self.model_path = model_path
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        if model is None:
            self.model = YOLO(model_path)  # Load the model from the specified path
        else:
            self.model = model
        self.elements = elements
        self.conf_dict = conf_dict or {}
        self.minimize_points = minimize_points

        self.use_box_propt_sam = use_box_propt_sam

        # model class index -> CVAT label name
        self.index_to_label = self._resolve_class_map(class_map or {})
        # Only run inference for model classes that are mapped
        self.classes = sorted(self.index_to_label)

        if self.use_box_propt_sam and self.segment:
            from ultralytics.models.fastsam import FastSAMPredictor

            # Create FastSAMPredictor
            overrides = dict(
                conf=0.15,
                task="segment",
                mode="predict",
                model="FastSAM-x.pt",
                save=False,
                imgsz=1024,
                verbose=False,
                iou=0.85
            )
            self.FastSAMPredictor = FastSAMPredictor(overrides=overrides)

            print(
                "Instance segmentation will be performed using a zero-shot approach "
                "thanks to feeding detected boxes through a pre-trained SAM network."
            )

    def _resolve_class_map(self, class_map):
        if not class_map:
            raise ValueError(
                "No classes configured. Add a 'mapping' section to the config file: "
                "model class name -> CVAT label name."
            )

        # Legacy mode: keys are model class indices from the YOLO weights file.
        if all(isinstance(key, int) for key in class_map):
            index_to_label = {int(key): str(value) for key, value in class_map.items()}
            unknown = sorted(set(index_to_label) - set(self.model.names))
            if unknown:
                raise ValueError(
                    f"Model '{self.model_path}' has no classes with indices {unknown}."
                )
            print(
                "WARNING: 'names' keys are treated as model class indices (legacy mode). "
                "Use a 'mapping' section with model class names instead."
            )
            return index_to_label

        name_to_index = {
            normalize_class_name(name): index for index, name in self.model.names.items()
        }
        index_to_label = {}
        unknown = []
        for model_name, label in class_map.items():
            index = name_to_index.get(normalize_class_name(model_name))
            if index is None:
                unknown.append(model_name)
            else:
                index_to_label[index] = str(label)

        if unknown:
            available = ", ".join(
                self.model.names[index] for index in sorted(self.model.names)[:20]
            )
            raise ValueError(
                f"Classes {unknown} are not present in model '{self.model_path}'. "
                f"Available classes (first 20): {available}"
            )
        return index_to_label

    def _conf_for(self, model_index):
        label = self.index_to_label[model_index]
        if label in self.conf_dict:
            return float(self.conf_dict[label])
        model_name = self.model.names[model_index]
        if model_name in self.conf_dict:
            return float(self.conf_dict[model_name])
        return self.conf

    def process(self):
        """
        Processes the dataset for inference.

        Returns:
            DataGen: Processed dataset with inference results.
        """
        for element in tqdm(self.elements, desc="Processing images", unit="image", total=len(self.elements)):
            predictions = self.model.predict(
                element.image,
                imgsz=self.imgsz,
                conf=self.conf,
                iou=self.iou,
                verbose=False,
                classes=self.classes,
                retina_masks=True,
            )[0]

            boxes = predictions.boxes
            if boxes is None or len(boxes) == 0:
                element.bbox = []
                element.category_names = []
                element.detected_masks = []
                element.areas = []
                element.iscrowd = 0
                continue

            xyxy = boxes.xyxy.cpu().float().tolist()
            confidences = boxes.conf.cpu().float().tolist()
            class_indices = boxes.cls.cpu().int().tolist()

            # Filter by the confidence threshold of each CVAT label
            kept = [
                i
                for i in range(len(class_indices))
                if confidences[i] >= self._conf_for(class_indices[i])
            ]

            element.bbox = [
                [xyxy[i][0], xyxy[i][1], xyxy[i][2] - xyxy[i][0], xyxy[i][3] - xyxy[i][1]]
                for i in kept
            ]
            element.category_names = [self.index_to_label[class_indices[i]] for i in kept]

            if self.segment:
                if self.use_box_propt_sam:
                    # process boxes as input prompt for sam
                    everything_results = self.FastSAMPredictor(element.image)
                    detected_masks = []

                    for box in element.bbox:
                        ann = self.FastSAMPredictor.prompt(
                            everything_results,
                            bboxes=[[
                                int(box[0]),
                                int(box[1]),
                                int(box[2] + box[0]),
                                int(box[3] + box[1]),
                            ]],
                        )[0]

                        if self.minimize_points:
                            detected_mask = self.minimize_contours(
                                ann.masks.data.cpu().numpy(), [0], element.image
                            )[0]
                        else:
                            detected_mask = ann.masks.xy[0].flatten()

                        detected_masks.append(detected_mask)

                    element.detected_masks = [
                        [float(value) for value in mask] for mask in detected_masks
                    ]
                    element.areas = [0 for _ in element.detected_masks]
                else:
                    masks = getattr(predictions, "masks", None)
                    if masks is None:
                        # Detection-only model: keep boxes, no polygons
                        element.detected_masks = []
                        element.areas = [box[2] * box[3] for box in element.bbox]
                    else:
                        if self.minimize_points:
                            element.detected_masks = self.minimize_contours(
                                masks.data.cpu().numpy(), kept, element.image
                            )
                        else:
                            element.detected_masks = [
                                [float(value) for value in masks.xy[i].flatten()]
                                for i in kept
                            ]
                        element.areas = [0 for _ in kept]
                element.iscrowd = 0
            else:
                element.detected_masks = []
                element.areas = [box[2] * box[3] for box in element.bbox]
        return self.elements

    def minimize_contours(self, predictions_masks_xy, filtered_indices, image):
        """
        Converts masks into minimized contours and saves points in the required format.

        Args:
            predictions_masks_xy (list): List of segments in pixel coordinates represented as tensors.
            filtered_indices (list): Indices of masks to process.
        Returns:
            list: List of minimized contours in the form of a list of points.
        """
        minimized_contours = []
        predictions_masks_xy = np.array(predictions_masks_xy)

        for i, mask in enumerate(predictions_masks_xy):
            if i in filtered_indices:
                mask_resized = cv2.resize(
                    np.array(mask),
                    (image.shape[1], image.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
                # Find all contours without hierarchy
                mask_contours, _ = cv2.findContours(
                    mask_resized.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
                )
                # Select the contour with the largest area
                max_contour = max(mask_contours, key=cv2.contourArea)
                # Simplify the contour
                epsilon = 0.002 * cv2.arcLength(max_contour, True)  # Set the approximation accuracy
                approx = cv2.approxPolyDP(
                    max_contour, epsilon, True
                )  # Get the approximated contour
                # Convert all coordinate values to integers
                mask_contour_int = list(map(int, approx.reshape(-1)))
                minimized_contours.append(mask_contour_int)
        return minimized_contours
