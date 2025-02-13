#from lib.providers.commands import command
import json
import os
import replicate
from PIL import Image
import io
import asyncio

async def yolo(img_path: str, class_names: str):
    input_media = open(img_path, "rb")
    output = replicate.run(
        "franz-biz/yolo-world-xl:fd1305d3fc19e81540542f51c2530cf8f393e28cc6ff4976337c3e2b75c7c292",
        input={
            "nms_thr": 0.5,
            "score_thr": 0.05,
            "class_names": class_names,
            "input_media": input_media,
            "return_json": True,
            "max_num_boxes": 100
        }
    )
    print(output)
    return output

def extract_subimages(image_path: str, detections_json: str) -> tuple:
    """Extract subimages from the original image based on detection bounding boxes.
    
    Args:
        image_path: Path to the original image
        detections_json: JSON string containing detection results
        
    Returns:
        Tuple of (list of PIL Image objects, list of dicts with image dimensions)
    """
    # Load the original image
    original_image = Image.open(image_path)
    
    # Parse the detections JSON
    detections = json.loads(detections_json)
    
    # Extract each subimage
    subimages = []
    dimensions = []
    for det_key, det in detections.items():
        # Extract coordinates
        box = (int(det['x0']), int(det['y0']), int(det['x1']), int(det['y1']))
        # Crop the subimage
        subimage = original_image.crop(box)
        # Calculate dimensions
        width = int(det['x1']) - int(det['x0'])
        height = int(det['y1']) - int(det['y0'])
        dimensions.append({
            'width': width,
            'height': height
        })
        subimages.append(subimage)
    return subimages, dimensions

#@command()
async def get_object_images(abs_image_file_path: str, class_names: str, context=None):
    """Uses YOLO to extract rectangular images of objects found in an input image.
    
    abs_image_file_path - The path to the input image to do bounding box object detection in.
    class_names - A comma-separated list of object class names
    
    Returns:

    A list of absolute file paths to images with the extracted objects.
    
    Example:
    
    { "get_object_images": {
        "abs_image_file_path": "/path/to/image_to_analyze.png",
        "class_names": "dog, cat, pet"
      }
    }

    """
    # Run YOLO detection
    output = await yolo(abs_image_file_path, class_names)
    
    # Extract subimages
    subimages, dimensions = extract_subimages(abs_image_file_path, output['json_str'])
    
    # Save subimages to files
    results = []
    for i, img in enumerate(subimages):
        base_path = os.path.splitext(os.path.abspath(abs_image_file_path))[0]
        output_path = f"{base_path}_obj_{i}.jpg"
        img.save(output_path, "JPEG")
        results.append({
            'path': output_path,
            'width': dimensions[i]['width'],
            'height': dimensions[i]['height']
        })
    print()
    print(results)
    return results

if __name__ == "__main__":
    asyncio.run(get_object_images("baseballcards.jpg", "baseball card, trading card, card"))
    
