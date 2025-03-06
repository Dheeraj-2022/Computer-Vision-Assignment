import os
import cv2
import numpy as np
import argparse
from pathlib import Path
from ultralytics import YOLO


def crop_person(image, person_box, margin=0):
    """
    Crop a person from the image using bounding box coordinates
    
    Parameters:
    image (numpy.ndarray): Original image
    person_box (list): Person bounding box [x1, y1, x2, y2]
    margin (int): Optional margin to add around the person
    
    Returns:
    tuple: (cropped_image, crop_info)
    """
    x1, y1, x2, y2 = [int(coord) for coord in person_box]
    
    # Add margin if specified
    if margin > 0:
        height, width = image.shape[:2]
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(width, x2 + margin)
        y2 = min(height, y2 + margin)
    
    # Crop image
    cropped_image = image[y1:y2, x1:x2]
    
    # Store crop information for later mapping coordinates back
    crop_info = {'x1': x1, 'y1': y1, 'width': x2 - x1, 'height': y2 - y1}
    
    return cropped_image, crop_info


def map_to_original_coords(ppe_box, crop_info):
    """
    Map PPE detection coordinates from cropped image back to original image
    
    Parameters:
    ppe_box (list): PPE bounding box in cropped image [x1, y1, x2, y2]
    crop_info (dict): Information about the crop
    
    Returns:
    list: PPE bounding box in original image coordinates
    """
    x1, y1, x2, y2 = ppe_box
    
    # Map coordinates back to original image
    original_x1 = crop_info['x1'] + x1
    original_y1 = crop_info['y1'] + y1
    original_x2 = crop_info['x1'] + x2
    original_y2 = crop_info['y1'] + y2
    
    return [original_x1, original_y1, original_x2, original_y2]


def draw_detection(image, box, class_name, confidence, color=(0, 255, 0), thickness=2):
    """
    Draw a detection bounding box and label on the image
    
    Parameters:
    image (numpy.ndarray): Image to draw on
    box (list): Bounding box coordinates [x1, y1, x2, y2]
    class_name (str): Class name of the detection
    confidence (float): Confidence score
    color (tuple): BGR color for the box
    thickness (int): Line thickness
    
    Returns:
    numpy.ndarray: Image with drawn detection
    """
    x1, y1, x2, y2 = [int(coord) for coord in box]
    
    # Draw bounding box
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    
    # Prepare label text with confidence
    label = f"{class_name} {confidence:.2f}"
    
    # Get size of the text for the background rectangle
    (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    
    # Draw background rectangle for text
    cv2.rectangle(image, (x1, y1 - text_height - 5), (x1 + text_width, y1), color, -1)
    
    # Draw text
    cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    return image


def process_image(image_path, output_path, person_model, ppe_model, person_conf=0.5, ppe_conf=0.5):
    """
    Process an image through both person and PPE detection models
    
    Parameters:
    image_path (str): Path to the input image
    output_path (str): Path to save the output image
    person_model (YOLO): Person detection model
    ppe_model (YOLO): PPE detection model
    person_conf (float): Confidence threshold for person detection
    ppe_conf (float): Confidence threshold for PPE detection
    """
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return
    
    # Make a copy for results
    result_image = image.copy()
    
    # Person detection
    person_results = person_model(image, conf=person_conf, verbose=False)[0]
    
    # Process each detected person
    for det in person_results.boxes.data:
        # Extract person box and confidence
        x1, y1, x2, y2, conf, cls = det
        person_box = [x1, y1, x2, y2]
        person_conf = conf
        
        # Get class name for person
        person_cls_name = person_results.names[int(cls)]
        
        # Only process if it's a person
        if person_cls_name == 'person':
            # Draw person bounding box (blue color)
            draw_detection(result_image, person_box, person_cls_name, person_conf, color=(255, 0, 0))
            
            # Crop person for PPE detection
            cropped_person, crop_info = crop_person(image, person_box, margin=5)
            
            # Skip if cropped image is too small
            if cropped_person.size == 0 or cropped_person.shape[0] <= 0 or cropped_person.shape[1] <= 0:
                continue
            
            # PPE detection on cropped person
            ppe_results = ppe_model(cropped_person, conf=ppe_conf, verbose=False)[0]
            
            # Process each detected PPE item
            for ppe_det in ppe_results.boxes.data:
                # Extract PPE box and confidence
                ppe_x1, ppe_y1, ppe_x2, ppe_y2, ppe_conf, ppe_cls = ppe_det
                ppe_box = [ppe_x1, ppe_y1, ppe_x2, ppe_y2]
                
                # Get class name for PPE
                ppe_cls_name = ppe_results.names[int(ppe_cls)]
                
                # Map PPE coordinates back to original image
                original_ppe_box = map_to_original_coords(ppe_box, crop_info)
                
                # Draw PPE bounding box (green color)
                draw_detection(result_image, original_ppe_box, ppe_cls_name, ppe_conf, color=(0, 255, 0))
    
    # Save result
    cv2.imwrite(output_path, result_image)
    print(f"Processed image saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Run person and PPE detection inference')
    parser.add_argument('input_dir', type=str, help='Directory containing input images')
    parser.add_argument('output_dir', type=str, help='Directory to save output images')
    parser.add_argument('person_det_model', type=str, help='Path to person detection model weights')
    parser.add_argument('ppe_detection_model', type=str, help='Path to PPE detection model weights')
    
    args = parser.parse_args()
    
    # Setup paths
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load models
    try:
        print(f"Loading person detection model from {args.person_det_model}")
        person_model = YOLO(args.person_det_model)
        
        print(f"Loading PPE detection model from {args.ppe_detection_model}")
        ppe_model = YOLO(args.ppe_detection_model)
    except Exception as e:
        print(f"Error loading models: {e}")
        return
    
    # Process images
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    image_files = [f for f in input_dir.glob('*') if f.suffix.lower() in image_extensions]
    
    print(f"Found {len(image_files)} images to process")
    
    for image_file in image_files:
        output_path = output_dir / image_file.name
        print(f"Processing {image_file}")
        process_image(str(image_file), str(output_path), person_model, ppe_model)
    
    print("Inference complete!")


if __name__ == "__main__":
    main()