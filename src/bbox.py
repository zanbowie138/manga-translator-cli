"""
Bounding box utility functions for working with detection boxes.
All boxes are in format [x1, y1, x2, y2] (xyxy format).
"""


def is_box_inside(box1, box2, threshold=0.0):
    """
    Check if box1 is inside box2 (or vice versa) with optional threshold for touching
    
    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]
        threshold: Distance threshold for considering boxes as touching
    
    Returns:
        True if one box is inside the other or they're touching
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Check if box1 is inside box2 (with threshold)
    if (x1_1 >= x1_2 - threshold and y1_1 >= y1_2 - threshold and
        x2_1 <= x2_2 + threshold and y2_1 <= y2_2 + threshold):
        return True
    
    # Check if box2 is inside box1 (with threshold)
    if (x1_2 >= x1_1 - threshold and y1_2 >= y1_1 - threshold and
        x2_2 <= x2_1 + threshold and y2_2 <= y2_1 + threshold):
        return True
    
    return False


def are_boxes_touching(box1, box2, threshold=10):
    """
    Check if two boxes are touching or overlapping
    
    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]
        threshold: Pixel threshold for considering boxes as touching
    
    Returns:
        True if boxes are touching or overlapping
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Check for overlap or proximity
    # Boxes overlap if: x1_1 < x2_2 + threshold and x2_1 + threshold > x1_2
    #                   and y1_1 < y2_2 + threshold and y2_1 + threshold > y1_2
    overlap_x = x1_1 < x2_2 + threshold and x2_1 + threshold > x1_2
    overlap_y = y1_1 < y2_2 + threshold and y2_1 + threshold > y1_2
    
    return overlap_x and overlap_y


def merge_boxes(box1, box2):
    """
    Merge two boxes into a single bounding box
    
    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]
    
    Returns:
        Merged box as [x1, y1, x2, y2] (union of both boxes)
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Take the union (min of mins, max of maxes)
    merged = [
        min(x1_1, x1_2),
        min(y1_1, y1_2),
        max(x2_1, x2_2),
        max(y2_1, y2_2)
    ]
    return merged


def box_contains(box1, box2, threshold=0.0):
    """
    Check if box1 strictly contains box2 (box2 is inside box1)
    
    Args:
        box1: [x1, y1, x2, y2] - potential parent
        box2: [x1, y1, x2, y2] - potential child
        threshold: Distance threshold for considering boxes
    
    Returns:
        True if box1 contains box2
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # box1 contains box2 if box2 is completely inside box1
    return (x1_2 >= x1_1 - threshold and y1_2 >= y1_1 - threshold and
            x2_2 <= x2_1 + threshold and y2_2 <= y2_1 + threshold)


def remove_parent_boxes(boxes, threshold=0.0):
    """
    Remove parent bounding boxes from compound speech bubbles and keep children.
    For compound speech bubbles (bubbles within bubbles), removes the parent (outer) 
    box and keeps the child (inner) box.
    
    Args:
        boxes: List of bounding boxes as [x1, y1, x2, y2]
        threshold: Distance threshold for containment check
    
    Returns:
        List of bounding boxes with parent boxes removed
    """
    if not boxes:
        return boxes
    
    # Create a copy to work with
    remaining = boxes.copy()
    filtered = []
    
    while remaining:
        current_bbox = remaining.pop(0)
        is_parent = False
        
        # Check if current box contains any other box (it's a parent)
        for other_bbox in remaining:
            if box_contains(current_bbox, other_bbox, threshold=threshold):
                is_parent = True
                break
        
        # Only keep boxes that are not parents
        if not is_parent:
            filtered.append(current_bbox)
        else:
            print(f"  Removed parent box from compound speech bubble, keeping child box")
    
    return filtered


def combine_overlapping_bubbles(boxes, touch_threshold=10):
    """
    Combine compound speech bubble bounding boxes that are inside one another or touching.
    Merges overlapping or adjacent bubbles from compound speech bubbles into single boxes.
    
    Args:
        boxes: List of bounding boxes as [x1, y1, x2, y2]
        touch_threshold: Pixel threshold for considering boxes as touching
    
    Returns:
        List of merged bounding boxes as [x1, y1, x2, y2]
    """
    if not boxes:
        return boxes
    
    # Create a copy to work with
    remaining = boxes.copy()
    merged = []
    
    while remaining:
        # Start with the first box
        current_bbox = remaining.pop(0)
        
        # Find all boxes that are inside or touching the current box
        to_merge = []
        i = 0
        while i < len(remaining):
            other_bbox = remaining[i]
            
            # Check if boxes are inside each other or touching
            if is_box_inside(current_bbox, other_bbox, threshold=touch_threshold) or \
               are_boxes_touching(current_bbox, other_bbox, threshold=touch_threshold):
                to_merge.append(other_bbox)
                remaining.pop(i)
            else:
                i += 1
        
        # Merge all boxes
        if to_merge:
            # Merge bounding boxes
            for other_bbox in to_merge:
                current_bbox = merge_boxes(current_bbox, other_bbox)
            
            merged.append(current_bbox)
            print(f"  Merged {len(to_merge) + 1} overlapping/touching compound speech bubbles into one")
        else:
            # No merging needed, keep as is
            merged.append(current_bbox)
    
    return merged

