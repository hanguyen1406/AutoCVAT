import json


class COCOConverter:
    """Class for converting detection data to COCO format (Common Objects in Context).

    Attributes:
        elements (list): List of elements containing detection information.
        names (dict): CVAT label id -> CVAT label name.
    """

    def __init__(self, elements, names):
        """Initialization of the COCOConverter object.

        Args:
            elements (list): List of elements containing detection information.
            names (dict): CVAT label id -> CVAT label name.
        """
        self.elements = elements
        self.names = {int(key): str(value) for key, value in names.items()}
        # COCO category ids must be positive; CVAT matches annotations by name
        self.name_to_id = {}
        for key, name in self.names.items():
            self.name_to_id.setdefault(name, key + 1)

    def convert_to_coco(self):
        """Converts the data to COCO format.

        Returns:
            str: JSON string representing the data in COCO format.
        """
        # Creating a list of categories
        categories = [
            {"id": category_id, "name": name, "supercategory": ""}
            for name, category_id in self.name_to_id.items()
        ]

        # Creating a list of images
        images = [
            {
                "id": elem.image_id,
                "width": elem.width,
                "height": elem.height,
                "file_name": elem.file_name,
                "license": 0,
                "flickr_url": "",
                "coco_url": "",
                "date_captured": 0,
            }
            for elem in self.elements
        ]

        # Creating a list of annotations
        annotations = []
        annotation_id = 1
        for elem in self.elements:

            for counter, (bbox, area, category_name) in enumerate(
                zip(elem.bbox, elem.areas, elem.category_names)
            ):
                annotation = {
                    "id": annotation_id,
                    "image_id": elem.image_id,
                    "category_id": self.name_to_id[category_name],
                    "segmentation": [],
                    "bbox": [float(value) for value in bbox],
                    "area": float(area),
                    "iscrowd": elem.iscrowd,
                }
                if counter < len(elem.detected_masks) and elem.detected_masks[counter]:
                    annotation["segmentation"] = [
                        [float(value) for value in elem.detected_masks[counter]]
                    ]
                    annotation["attributes"] = {"occluded": False}
                else:
                    annotation["attributes"] = {"occluded": False, "rotation": 0}
                annotations.append(annotation)
                annotation_id += 1

        # Creating the COCO dictionary
        coco_data = {
            "licenses": [{"name": "", "id": 0, "url": ""}],
            "info": {
                "contributor": "",
                "date_created": "",
                "description": "",
                "url": "",
                "version": "",
                "year": "",
            },
            "categories": categories,
            "images": images,
            "annotations": annotations,
        }

        # Convert to JSON string
        coco_json = json.dumps(coco_data)

        return coco_json
