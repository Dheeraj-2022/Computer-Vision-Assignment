import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
import argparse
from pathlib import Path
import shutil
import random
from collections import Counter
import matplotlib.pyplot as plt


def create_dataset_structure(base_dir):
    """
    Create necessary directories for YOLOv8 dataset
    
    Parameters:
    base_dir (Path): Base directory path
    
    Returns:
    dict: Dictionary containing paths
    """
    # Create main directories
    paths = {
        'person': {
            'base': base_dir / 'person_detection',
            'train': base_dir / 'person_detection/train',
            'val': base_dir / 'person_detection/val',
            'test': base_dir / 'person_detection/test'
        },
        'ppe': {
            'base': base_dir / 'ppe_detection',
            'train': base_dir / 'ppe_detection/train',
            'val': base_dir / 'ppe_detection/val',
            'test': base_dir / 'ppe_detection/test'
        }
    }
    
    # Create subdirectories
    for dataset_type in ['person', 'ppe']:
        for split in ['train', 'val', 'test']:
            os.makedirs(paths[dataset_type][split] / 'images', exist_ok=True)
            os.makedirs(paths[dataset_type][split] / 'labels', exist_ok=True)
    
    return paths


def read_xml_annotation(xml_path):
    """
    Read PascalVOC XML annotation file
    
    Parameters:
    xml_path (str): Path to XML annotation file
    
    Returns:
    tuple: (image_width, image_height, list of objects)
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Get image size
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    
    # Get objects
    objects = []
    for obj in root.findall('object'):
        name = obj.find('name').text.strip()
        
        bbox = obj.find('bndbox')
        xmin = float(bbox.find('xmin').text)
        ymin = float(bbox.find('ymin').text)
        xmax = float(bbox.find('xmax').text)
        ymax = float(bbox.find('ymax').text)
        
        objects.append({
            'name': name,
            'bbox': [xmin, ymin, xmax, ymax]
        })
    
    return width, height, objects


def convert_to_yolo_format(size, box):
    """
    Convert bounding box from VOC format to YOLO format
    
    Parameters:
    size (tuple): (width, height) of the image
    box (list): [xmin, ymin, xmax, ymax] coordinates
    
    Returns:
    tuple: (x_center, y_center, width, height) normalized coordinates
    """
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    
    xmin, ymin, xmax, ymax = box
    
    # Calculate center, width, height
    x_center = (xmin + xmax) / 2.0
    y_center = (ymin + ymax) / 2.0
    w = xmax - xmin
    h = ymax - ymin
    
    # Normalize
    x_center *= dw
    y_center *= dh
    w *= dw
    h *= dh
    
    return x_center, y_center, w, h


def crop_person_and_update_annotations(image, image_width, image_height, person_bbox, objects, class_map, margin=10):
    """
    Crop a person from image and update annotations for PPE items
    
    Parameters:
    image (numpy.ndarray): Original image
    image_width (int): Original image width
    image_height (int): Original image height
    person_bbox (list): Person bounding box [xmin, ymin, xmax, ymax]
    objects (list): List of all objects in the image
    class_map (dict): Mapping of class names to class indices
    margin (int): Margin to add around person crop
    
    Returns:
    tuple: (cropped_image, updated_annotations_list)
    """
    # Extract person coordinates with margin
    xmin, ymin, xmax, ymax = person_bbox
    
    # Add margin
    xmin = max(0, xmin - margin)
    ymin = max(0, ymin - margin)
    xmax = min(image_width, xmax + margin)
    ymax = min(image_height, ymax + margin)
    
    # Crop the image
    cropped_image = image[int(ymin):int(ymax), int(xmin):int(xmax)]
    
    # Calculate new width and height
    crop_width = xmax - xmin
    crop_height = ymax - ymin
    
    # Find PPE items that overlap with the person and update coordinates
    updated_annotations = []
    
    for obj in objects:
        if obj['name'] == 'person':
            continue  # Skip persons for PPE dataset
        
        # Get PPE bbox
        ppe_xmin, ppe_ymin, ppe_xmax, ppe_ymax = obj['bbox']
        
        # Check if PPE item overlaps with the person
        if (ppe_xmax > xmin and ppe_xmin < xmax and
            ppe_ymax > ymin and ppe_ymin < ymax):
            
            # Adjust coordinates to the cropped image
            new_xmin = max(0, ppe_xmin - xmin)
            new_ymin = max(0, ppe_ymin - ymin)
            new_xmax = min(crop_width, ppe_xmax - xmin)
            new_ymax = min(crop_height, ppe_ymax - ymin)
            
            # Only include if the bbox is still valid
            if new_xmax > new_xmin and new_ymax > new_ymin:
                # Convert to YOLO format
                x_center, y_center, w, h = convert_to_yolo_format(
                    (crop_width, crop_height),
                    [new_xmin, new_ymin, new_xmax, new_ymax]
                )
                
                # Get class ID
                class_id = class_map.get(obj['name'])
                
                if class_id is not None:
                    updated_annotations.append({
                        'class_id': class_id,
                        'x_center': x_center,
                        'y_center': y_center,
                        'width': w,
                        'height': h
                    })
    
    return cropped_image, updated_annotations


def prepare_data(source_dir, output_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, visualize_distribution=True):
    """
    Prepare datasets for both person and PPE detection
    
    Parameters:
    source_dir (Path): Source directory with images and annotations
    output_dir (Path): Output directory for prepared datasets
    train_ratio, val_ratio, test_ratio (float): Data split ratios
    visualize_distribution (bool): Whether to visualize class distribution
    """
    # Read class mapping
    class_map = {}
    class_file = source_dir / 'classes.txt'
    with open(class_file, 'r') as f:
        for idx, line in enumerate(f.read().splitlines()):
            if line.strip():
                class_map[line.strip()] = idx
    
    print(f"Class mapping: {class_map}")
    
    # Create dataset structure
    paths = create_dataset_structure(output_dir)
    
    # Get all XML annotation files
    annotation_files = list(source_dir.glob('annotations/*.xml'))
    random.shuffle(annotation_files)
    
    # Calculate split sizes
    total_files = len(annotation_files)
    train_size = int(total_files * train_ratio)
    val_size = int(total_files * val_ratio)
    test_size = total_files - train_size - val_size
    
    # Split the data
    train_files = annotation_files[:train_size]
    val_files = annotation_files[train_size:train_size+val_size]
    test_files = annotation_files[train_size+val_size:]
    
    print(f"Data split: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")
    
    # Counters for class distribution
    person_class_counter = Counter()
    ppe_class_counter = Counter()
    ppe_items_per_person = []
    
    # Process each annotation file
    for split_name, files in [('train', train_files), ('val', val_files), ('test', test_files)]:
        print(f"Processing {split_name} split...")
        
        for xml_file in files:
            # Get corresponding image path
            image_name = xml_file.stem + '.jpg'
            image_path = source_dir / 'images' / image_name
            
            if not image_path.exists():
                print(f"Warning: Image {image_name} not found. Skipping...")
                continue
            
            # Read image
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"Warning: Could not read image {image_path}. Skipping...")
                continue
            
            # Read annotation
            try:
                image_width, image_height, objects = read_xml_annotation(xml_file)
            except Exception as e:
                print(f"Error reading annotation {xml_file}: {e}. Skipping...")
                continue
            
            # Process for person detection (whole image)
            person_yolo_labels = []
            persons_in_image = []
            
            for obj in objects:
                class_id = class_map.get(obj['name'])
                if class_id is not None:
                    # Convert to YOLO format
                    xmin, ymin, xmax, ymax = obj['bbox']
                    x_center, y_center, w, h = convert_to_yolo_format(
                        (image_width, image_height),
                        [xmin, ymin, xmax, ymax]
                    )
                    
                    # Store person annotations
                    person_yolo_labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")
                    
                    # Collect person objects for later cropping
                    if obj['name'] == 'person':
                        persons_in_image.append(obj['bbox'])
                        person_class_counter['person'] += 1
            
            # Save person detection annotation
            person_label_path = paths['person'][split_name] / 'labels' / (xml_file.stem + '.txt')
            with open(person_label_path, 'w') as f:
                f.write('\n'.join(person_yolo_labels))
            
            # Copy image for person detection dataset
            shutil.copy(image_path, paths['person'][split_name] / 'images' / image_name)
            
            # Process for PPE detection (cropped person images)
            for idx, person_bbox in enumerate(persons_in_image):
                # Crop person and update PPE annotations
                cropped_image, ppe_annotations = crop_person_and_update_annotations(
                    image, image_width, image_height, person_bbox, objects, class_map
                )
                
                # Skip if no PPE items found
                if not ppe_annotations:
                    continue
                
                # Count PPE items per person
                ppe_items_per_person.append(len(ppe_annotations))
                
                # Save cropped image
                cropped_image_name = f"{xml_file.stem}_person{idx}.jpg"
                cropped_image_path = paths['ppe'][split_name] / 'images' / cropped_image_name
                cv2.imwrite(str(cropped_image_path), cropped_image)
                
                # Save PPE annotations
                ppe_label_path = paths['ppe'][split_name] / 'labels' / (xml_file.stem + f"_person{idx}.txt")
                with open(ppe_label_path, 'w') as f:
                    for ann in ppe_annotations:
                        f.write(f"{ann['class_id']} {ann['x_center']:.6f} {ann['y_center']:.6f} {ann['width']:.6f} {ann['height']:.6f}\n")
                        ppe_class_counter[list(class_map.keys())[ann['class_id']]] += 1
    
    # Create data.yaml files for both datasets
    create_yaml_file(paths['person']['base'], 'person', class_map)
    create_yaml_file(paths['ppe']['base'], 'ppe', class_map)
    
    # Visualize class distributions
    if visualize_distribution:
        visualize_class_distribution(person_class_counter, 'Person Detection Classes', output_dir / 'person_class_distribution.png')
        visualize_class_distribution(ppe_class_counter, 'PPE Detection Classes', output_dir / 'ppe_class_distribution.png')
        
        plt.figure(figsize=(10, 6))
        plt.hist(ppe_items_per_person, bins=range(max(ppe_items_per_person) + 2), alpha=0.7)
        plt.title('Number of PPE Items per Person')
        plt.xlabel('Number of PPE Items')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        plt.savefig(output_dir / 'ppe_items_per_person.png')
        
        print(f"Class distribution visualizations saved to {output_dir}")


def create_yaml_file(output_path, dataset_name, class_map):
    """
    Create data.yaml file for YOLOv8 training
    
    Parameters:
    output_path (Path): Path to save the yaml file
    dataset_name (str): Name of the dataset
    class_map (dict): Mapping of class names to class indices
    """
    # Sort classes by index
    classes = [k for k, v in sorted(class_map.items(), key=lambda item: item[1])]
    
    # For PPE dataset, filter out 'person' class if needed
    if dataset_name == 'ppe' and 'person' in classes:
        filtered_classes = [c for c in classes if c != 'person']
        classes = filtered_classes
    
    with open(output_path / 'data.yaml', 'w') as f:
        f.write(f"# YOLOv8 {dataset_name} dataset config\n")
        f.write(f"path: {output_path.resolve()}\n")
        f.write(f"train: train/images\n")
        f.write(f"val: val/images\n")
        f.write(f"test: test/images\n\n")
        f.write(f"nc: {len(classes)}\n")
        f.write(f"names: {classes}\n")
    
    print(f"Created data.yaml for {dataset_name} dataset at {output_path / 'data.yaml'}")