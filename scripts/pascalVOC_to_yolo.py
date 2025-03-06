import os
import xml.etree.ElementTree as ET
import argparse
from pathlib import Path


def convert_bbox_voc_to_yolo(size, box):
    """
    Convert bounding box from VOC format to YOLO format
    
    Parameters:
    size (tuple): (width, height) of the image
    box (tuple): (xmin, ymin, xmax, ymax) coordinates of the bounding box
    
    Returns:
    tuple: (x_center, y_center, width, height) normalized coordinates for YOLO
    """
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    
    # Extract coordinates
    xmin, ymin, xmax, ymax = box
    
    # Calculate center coordinates, width and height
    x_center = (xmin + xmax) / 2.0
    y_center = (ymin + ymax) / 2.0
    w = xmax - xmin
    h = ymax - ymin
    
    # Normalize to [0, 1]
    x_center *= dw
    y_center *= dh
    w *= dw
    h *= dh
    
    return (x_center, y_center, w, h)


def read_class_names(class_file):
    """
    Read class names from file
    
    Parameters:
    class_file (str): path to the class names file
    
    Returns:
    dict: mapping of class names to class indices
    """
    class_dict = {}
    with open(class_file, 'r') as f:
        for idx, line in enumerate(f.read().splitlines()):
            if line.strip():  # Skip empty lines
                class_dict[line.strip()] = idx
    
    return class_dict


def convert_annotation(xml_file, output_path, class_dict):
    """
    Convert PascalVOC XML annotation to YOLO format
    
    Parameters:
    xml_file (str): path to the XML annotation file
    output_path (str): path to save the YOLO annotation
    class_dict (dict): mapping of class names to class indices
    
    Returns:
    int: number of objects converted
    """
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)
        
        filename = os.path.splitext(os.path.basename(xml_file))[0] + '.txt'
        with open(os.path.join(output_path, filename), 'w') as out_file:
            
            count = 0
            for obj in root.findall('object'):
                name = obj.find('name').text.strip()
                
                if name in class_dict:
                    class_id = class_dict[name]
                    
                    bbox = obj.find('bndbox')
                    xmin = float(bbox.find('xmin').text)
                    ymin = float(bbox.find('ymin').text)
                    xmax = float(bbox.find('xmax').text)
                    ymax = float(bbox.find('ymax').text)
                    
                    # Convert to YOLO format
                    x_center, y_center, w, h = convert_bbox_voc_to_yolo((width, height), (xmin, ymin, xmax, ymax))
                    
                    # Write to output file
                    out_file.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")
                    count += 1
                else:
                    print(f"Warning: Class '{name}' not found in class dictionary. Skipping...")
            
            return count
    except Exception as e:
        print(f"Error processing {xml_file}: {e}")
        return 0


def main():
    parser = argparse.ArgumentParser(description='Convert PascalVOC annotations to YOLO format')
    parser.add_argument('input_dir', type=str, help='Input base directory containing images and annotations')
    parser.add_argument('output_dir', type=str, help='Output directory to save YOLO annotations')
    
    args = parser.parse_args()
    
    # Paths setup
    base_dir = Path(args.input_dir)
    annotations_dir = base_dir / 'annotations'
    classes_file = base_dir / 'classes.txt'
    
    output_dir = Path(args.output_dir)
    output_labels_dir = output_dir / 'labels'
    output_images_dir = output_dir / 'images'
    
    # Create output directories
    os.makedirs(output_labels_dir, exist_ok=True)
    os.makedirs(output_images_dir, exist_ok=True)
    
    # Read class names
    if not classes_file.exists():
        print(f"Error: Class file not found at {classes_file}")
        return
    
    class_dict = read_class_names(str(classes_file))
    print(f"Found {len(class_dict)} classes: {class_dict}")
    
    # Process annotations
    xml_files = list(annotations_dir.glob('*.xml'))
    print(f"Found {len(xml_files)} XML annotation files to process")
    
    total_converted = 0
    for xml_file in xml_files:
        count = convert_annotation(str(xml_file), str(output_labels_dir), class_dict)
        total_converted += count
    
    print(f"Conversion complete. Converted {total_converted} objects from {len(xml_files)} annotation files.")


if __name__ == "__main__":
    main()