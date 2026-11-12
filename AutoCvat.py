import random
import os
import shutil
import yaml
import json

import cv2
import numpy as np

# Library for creating CLI (Command Line Interface) interfaces
import click

# Custom modules and classes
from nodes.Datagen import DataGen
from nodes.Inference import Inferencer
from nodes.AnnotMaker import COCOConverter


def validate_mapping(raw_mapping, cvat_names, where):
    """Validate model class name -> CVAT label name mapping."""
    if not raw_mapping:
        raise ValueError(
            f"{where}: missing 'mapping' section (model class name -> CVAT label name)."
        )
    mapping = {str(key): str(value) for key, value in raw_mapping.items()}
    unknown_targets = sorted(set(mapping.values()) - cvat_names)
    if unknown_targets:
        raise ValueError(
            f"{where}: mapping targets {unknown_targets} are not declared in 'names': "
            f"{sorted(cvat_names)}"
        )
    return mapping


def parse_confs(section, all_conf, cvat_names, warn_unknown):
    """Return (per-label confidences, default confidence) for a config section."""
    if all_conf is not None:
        return {}, all_conf
    confs = {
        str(key): float(value) for key, value in (section.get("confs") or {}).items()
    }
    if warn_unknown:
        unknown_confs = sorted(set(confs) - cvat_names)
        if unknown_confs:
            print(
                f"WARNING: confidence keys {unknown_confs} are not CVAT label "
                f"names {sorted(cvat_names)}; they will be ignored."
            )
    conf = min(confs.values()) if confs else float(section.get("conf", 0.4))
    return confs, conf


def reset_annotations(elements):
    for element in elements:
        element.bbox = []
        element.category_names = []
        element.detected_masks = []
        element.areas = []
        element.iscrowd = 0


def build_exclusion_masks(elements):
    """Rasterize instance annotations so stuff regions can be cut around them."""
    masks = {}
    for element in elements:
        mask = np.zeros(element.image.shape[:2], dtype=np.uint8)
        if element.detected_masks:
            for points in element.detected_masks:
                pts = np.asarray(points, dtype=np.float32).reshape(-1, 2).astype(np.int32)
                cv2.fillPoly(mask, [pts], 1)
        else:
            for x, y, w, h in element.bbox:
                cv2.rectangle(
                    mask, (int(x), int(y)), (int(x + w), int(y + h)), 1, -1
                )
        masks[element.image_id] = mask
    return masks


def merge_coco_parts(parts):
    """Merge COCO dicts that were produced with the same categories/images."""
    merged = {
        "licenses": parts[0]["licenses"],
        "info": parts[0]["info"],
        "categories": parts[0]["categories"],
        "images": parts[0]["images"],
        "annotations": [],
    }
    annotation_id = 1
    for part in parts:
        for annotation in part["annotations"]:
            annotation = dict(annotation)
            annotation["id"] = annotation_id
            annotation_id += 1
            merged["annotations"].append(annotation)
    return merged


def generate_and_save_as_json_cvat_labels(original_classes, file_name="new_classes.json"):
    # Generate a random unique color for each class
    colors = ["#" + "".join(random.choices("0123456789ABCDEF", k=6)) for _ in original_classes]

    # Create a new list of dictionaries with random colors and indexes as id
    new_classes = []
    for idx, cls in enumerate(original_classes):
        cls_dict = {"name": cls, "id": idx, "color": colors[idx], "type": "any", "attributes": []}
        new_classes.append(cls_dict)

    # Save the new list to a file
    with open(file_name, "w") as file:
        json.dump(new_classes, file, indent=2)
    print(f"JSON for CVAT saved to file {file_name}")


@click.command()
@click.option(
    "--img_folder",
    default="img_folder",
    help="Folder with images (task from CVAT)",
    type=str,
)
@click.option(
    "--weights",
    default="yolov8m.pt",
    help="Path to the Yolo model weights file with .pt extension",
    type=str,
)
@click.option(
    "--annotations_zip",
    default="annotations",
    help="Name of the COCO CVAT annotation zip archive",
    type=str,
)
@click.option(
    "--yaml_pth",
    default="config.yaml",
    help="The path to configuration yaml file",
    type=str,
)
@click.option(
    "--save_images_to_annotations_zip",
    default=False,
    help="Whether to create a zip file with photos to upload to CVAT",
    type=bool,
)
@click.option(
    "--cvat_json",
    default=False,
    help="Should I create a json file with classes for CVAT",
    type=bool,
)
@click.option(
    "--all_conf",
    default=None,
    help="The confidence parameter for all classes, confidences from config don't use",
    type=float,
)
@click.option(
    "--zero_shot_segmentation",
    default=False,
    help="When set to True, it allows for zero-shot instance segmentation using SAM from any source detection network",
    type=bool,
)
def main(**kwargs):
    result_folder = kwargs["annotations_zip"]
    model_pth = kwargs["weights"]
    input_folder = kwargs["img_folder"]
    configs = kwargs["yaml_pth"]
    save_images_to_annotations_zip = bool(kwargs["save_images_to_annotations_zip"])
    cvat_json = bool(kwargs["cvat_json"])
    all_conf = kwargs["all_conf"]
    use_box_propt_sam = kwargs["zero_shot_segmentation"]

    # Check if the folder exists
    if not os.path.exists(input_folder):
        print(f"Folder '{input_folder}' does not exist.")
        return 

    # Check if the folder is empty
    if not os.listdir(input_folder):
        print(f"Folder '{input_folder}' is empty.")
        return  
        
    # Load data from YAML file
    with open(configs, "r") as yaml_file:
        configs = yaml.safe_load(yaml_file)

    # CVAT label id -> CVAT label name
    names = {int(key): str(value) for key, value in configs["names"].items()}
    cvat_names = set(names.values())
    backend = str(configs.get("backend", "yolo")).lower()

    if backend not in {"yolo", "semantic", "panoptic"}:
        raise ValueError(
            f"Unknown backend '{backend}'. Use 'yolo', 'semantic' or 'panoptic'."
        )

    semantic_cfg = configs.get("semantic") or {}
    instance_cfg = configs.get("instance") or {}

    if backend == "panoptic":
        semantic_mapping = validate_mapping(
            semantic_cfg.get("mapping"), cvat_names, "semantic section"
        )
        instance_mapping = validate_mapping(
            instance_cfg.get("mapping"), cvat_names, "instance section"
        )
        semantic_confs, semantic_conf = parse_confs(
            semantic_cfg, all_conf, cvat_names, warn_unknown=True
        )
        instance_confs, instance_conf = parse_confs(
            instance_cfg, all_conf, cvat_names, warn_unknown=True
        )
    else:
        # Mapping: model class name -> CVAT label name.
        # Name-based mapping is required because class indices depend on the weights file.
        mapping = configs.get("mapping")
        if mapping:
            mapping = validate_mapping(mapping, cvat_names, "config")
            dict_confs, conf = parse_confs(
                configs, all_conf, cvat_names, warn_unknown=True
            )
        elif backend == "semantic":
            raise ValueError(
                "The 'semantic' backend requires a 'mapping' section: "
                "model class name -> CVAT label name."
            )
        else:
            # Legacy mode: keys of 'names' are treated as model class indices
            mapping = {int(key): name for key, name in names.items()}
            dict_confs, conf = parse_confs(
                configs, all_conf, cvat_names, warn_unknown=False
            )

    # If the result folder already exists, delete it and create a new one
    if os.path.exists(result_folder):
        shutil.rmtree(result_folder)

    # Create a result folder for annotations and images for uploading to cvat
    os.mkdir(result_folder)
    os.mkdir(result_folder + "/annotations")

    if save_images_to_annotations_zip:
        files = os.listdir(input_folder)
        os.mkdir(result_folder + "/images")
        for file_name in files:
            source_file = os.path.join(input_folder, file_name)
            destination_file = os.path.join(result_folder + "/images", file_name)
            shutil.copy2(source_file, destination_file)  

    # Create a JSON string in COCO format
    datagen = DataGen(input_folder)
    elements = datagen.process()

    # Inference each photo
    if backend == "panoptic":
        from nodes.SemanticSegmenter import SemanticSegmenter

        # 1) Instance (things) with YOLO
        instance_model_path = instance_cfg.get("model") or model_pth
        instance = Inferencer(
            elements,
            segment=instance_cfg.get("segment", True),
            model_path=instance_model_path,
            class_map=instance_mapping,
            conf_dict=instance_confs,
            conf=instance_conf,
            imgsz=instance_cfg.get("imgsz", 640),
            iou=instance_cfg.get("iou", 0.8),
            minimize_points=instance_cfg.get(
                "minimize_points", configs.get("minimize_points", False)
            ),
            use_box_propt_sam=use_box_propt_sam,
        )
        elements = instance.process()
        instance_data = json.loads(COCOConverter(elements, names).convert_to_coco())
        # Stuff regions are cut around detected things to avoid overlaps
        exclusion_masks = build_exclusion_masks(elements)
        reset_annotations(elements)

        # 2) Semantic (stuff) with SegFormer
        segmenter = SemanticSegmenter(
            elements,
            class_map=semantic_mapping,
            model_path=semantic_cfg.get("model"),
            conf_dict=semantic_confs,
            conf=semantic_conf,
            minimize_points=semantic_cfg.get(
                "minimize_points", configs.get("minimize_points", False)
            ),
            min_area=semantic_cfg.get("min_area", 200),
            img_size=semantic_cfg.get("imgsz") or semantic_cfg.get("semantic_imgsz"),
            exclude_masks=exclusion_masks,
        )
        elements = segmenter.process()
        semantic_data = json.loads(COCOConverter(elements, names).convert_to_coco())
        results = json.dumps(merge_coco_parts([semantic_data, instance_data]))
    elif backend == "semantic":
        from nodes.SemanticSegmenter import SemanticSegmenter

        print("Using the 'semantic' backend: boxes, iou and zero-shot SAM are ignored.")
        inferencer = SemanticSegmenter(
            elements,
            class_map=mapping,
            model_path=configs.get("semantic_model"),
            conf_dict=dict_confs,
            conf=conf,
            minimize_points=configs.get("minimize_points", False),
            min_area=configs.get("min_area", 200),
            img_size=configs.get("semantic_imgsz"),
        )
        elements = inferencer.process()
        results = COCOConverter(elements, names).convert_to_coco()
    else:
        inferencer = Inferencer(
            elements,
            segment=configs.get("segment", False),
            model_path=model_pth,
            class_map=mapping,
            conf_dict=dict_confs,
            conf=conf,
            imgsz=configs.get("imgsz", 640),
            iou=configs.get("iou", 0.8),
            minimize_points=configs.get("minimize_points", False),
            use_box_propt_sam=use_box_propt_sam,
        )
        elements = inferencer.process()
        results = COCOConverter(elements, names).convert_to_coco()

    # Write JSON to file
    output_file_path = os.path.join(result_folder + "/annotations", "instances_default.json")

    with open(output_file_path, "w") as output_file:
        output_file.write(results)

    # Create a zip archive from the result folder
    shutil.make_archive(result_folder, "zip", result_folder)

    # Print the path to the result folder in the terminal
    print(f"Annotations are located at: {result_folder}.zip")

    # Delete the result folder
    shutil.rmtree(result_folder)

    if cvat_json:
        generate_and_save_as_json_cvat_labels(list(names.values()))


if __name__ == "__main__":
    main()
