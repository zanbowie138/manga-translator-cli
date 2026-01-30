"""
Bounding box utility functions for working with detection boxes.
All boxes are in format [x1, y1, x2, y2] (xyxy format).
"""

from typing import List, Tuple, Union, Optional


class BoundingBox:
    """
    A bounding box class representing a rectangular region.
    Coordinates are in format (x1, y1, x2, y2) where x1 < x2 and y1 < y2.
    """
    
    def __init__(self, x1: float, y1: float, x2: float, y2: float):
        """
        Initialize a bounding box.
        
        Args:
            x1: Left coordinate
            y1: Top coordinate
            x2: Right coordinate
            y2: Bottom coordinate
        """
        # Normalize to ensure x1 < x2 and y1 < y2
        self.x1 = min(x1, x2)
        self.y1 = min(y1, y2)
        self.x2 = max(x1, x2)
        self.y2 = max(y1, y2)
    
    @classmethod
    def from_list(cls, box: List[float]) -> 'BoundingBox':
        """Create a BoundingBox from a list [x1, y1, x2, y2]."""
        return cls(box[0], box[1], box[2], box[3])
    
    @classmethod
    def from_tuple(cls, box: Tuple[float, ...]) -> 'BoundingBox':
        """Create a BoundingBox from a tuple (x1, y1, x2, y2)."""
        return cls(box[0], box[1], box[2], box[3])
    
    def to_list(self) -> List[float]:
        """Convert to list [x1, y1, x2, y2]."""
        return [self.x1, self.y1, self.x2, self.y2]
    
    def to_tuple(self) -> Tuple[float, float, float, float]:
        """Convert to tuple (x1, y1, x2, y2)."""
        return (self.x1, self.y1, self.x2, self.y2)
    
    def __iter__(self):
        """Allow unpacking: x1, y1, x2, y2 = bbox"""
        return iter([self.x1, self.y1, self.x2, self.y2])
    
    def __getitem__(self, index: int) -> float:
        """Allow indexing: bbox[0] = x1, bbox[1] = y1, etc."""
        return [self.x1, self.y1, self.x2, self.y2][index]
    
    def __len__(self) -> int:
        """Return length (always 4)."""
        return 4
    
    def __hash__(self) -> int:
        """Make hashable for use as dictionary keys."""
        return hash((self.x1, self.y1, self.x2, self.y2))
    
    def __eq__(self, other) -> bool:
        """Check equality."""
        if not isinstance(other, BoundingBox):
            return False
        return (self.x1 == other.x1 and self.y1 == other.y1 and
                self.x2 == other.x2 and self.y2 == other.y2)
    
    def __repr__(self) -> str:
        """String representation."""
        return f"BoundingBox(x1={self.x1}, y1={self.y1}, x2={self.x2}, y2={self.y2})"
    
    def is_valid(self) -> bool:
        """Check if the box is valid (x2 > x1 and y2 > y1)."""
        return self.x2 > self.x1 and self.y2 > self.y1
    
    def clip(self, image_width: int, image_height: int) -> 'BoundingBox':
        """
        Clip the bounding box to image bounds.
        
        Args:
            image_width: Width of the image
            image_height: Height of the image
        
        Returns:
            New BoundingBox clipped to image bounds.
        """
        x1 = max(0, int(self.x1))
        y1 = max(0, int(self.y1))
        x2 = min(image_width, int(self.x2))
        y2 = min(image_height, int(self.y2))
        return BoundingBox(x1, y1, x2, y2)
    
    def width(self) -> float:
        """Get the width of the bounding box."""
        return self.x2 - self.x1
    
    def height(self) -> float:
        """Get the height of the bounding box."""
        return self.y2 - self.y1
    
    def area(self) -> float:
        """Get the area of the bounding box."""
        return self.width() * self.height()
    
    def is_inside(self, other: 'BoundingBox', threshold: float = 0.0) -> bool:
        """
        Check if this box is inside another box (or vice versa) with optional threshold.
        
        Args:
            other: Another BoundingBox
            threshold: Distance threshold for considering boxes as touching
        
        Returns:
            True if one box is inside the other or they're touching
        """
        # Check if self is inside other (with threshold)
        if (self.x1 >= other.x1 - threshold and self.y1 >= other.y1 - threshold and
            self.x2 <= other.x2 + threshold and self.y2 <= other.y2 + threshold):
            return True
        
        # Check if other is inside self (with threshold)
        if (other.x1 >= self.x1 - threshold and other.y1 >= self.y1 - threshold and
            other.x2 <= self.x2 + threshold and other.y2 <= self.y2 + threshold):
            return True
        
        return False
    
    def is_touching(self, other: 'BoundingBox', threshold: float = 10.0) -> bool:
        """
        Check if two boxes are touching or overlapping.
        
        Args:
            other: Another BoundingBox
            threshold: Pixel threshold for considering boxes as touching
        
        Returns:
            True if boxes are touching or overlapping
        """
        overlap_x = self.x1 < other.x2 + threshold and self.x2 + threshold > other.x1
        overlap_y = self.y1 < other.y2 + threshold and self.y2 + threshold > other.y1
        return overlap_x and overlap_y
    
    def contains(self, other: 'BoundingBox', threshold: float = 0.0) -> bool:
        """
        Check if this box strictly contains another box.
        
        Args:
            other: Another BoundingBox (potential child)
            threshold: Distance threshold for growing this box (other remains unchanged)
        
        Returns:
            True if this box (grown by threshold) contains the other box
        """
        # Grow this box by threshold on all sides
        grown_x1 = self.x1 - threshold
        grown_y1 = self.y1 - threshold
        grown_x2 = self.x2 + threshold
        grown_y2 = self.y2 + threshold
        
        # Check if other box is completely inside the grown box
        return (other.x1 >= grown_x1 and other.y1 >= grown_y1 and
                other.x2 <= grown_x2 and other.y2 <= grown_y2)
    
    def merge(self, other: 'BoundingBox') -> 'BoundingBox':
        """
        Merge this box with another box.
        
        Args:
            other: Another BoundingBox
        
        Returns:
            New BoundingBox that is the union of both boxes
        """
        return BoundingBox(
            min(self.x1, other.x1),
            min(self.y1, other.y1),
            max(self.x2, other.x2),
            max(self.y2, other.y2)
        )


# Convenience functions for backward compatibility and list operations

def clip_box(box: Union[List[float], BoundingBox], image_width: int, image_height: int) -> BoundingBox:
    """
    Clip a bounding box to image bounds.
    
    Args:
        box: Bounding box as [x1, y1, x2, y2] or BoundingBox instance
        image_width: Width of the image
        image_height: Height of the image
    
    Returns:
        BoundingBox clipped to image bounds.
    """
    if isinstance(box, BoundingBox):
        return box.clip(image_width, image_height)
    else:
        bbox = BoundingBox.from_list(box)
        return bbox.clip(image_width, image_height)


def validate_box(box: Union[List[float], BoundingBox]) -> bool:
    """
    Validate that a bounding box is valid (x2 > x1 and y2 > y1).
    
    Args:
        box: Bounding box as [x1, y1, x2, y2] or BoundingBox instance
    
    Returns:
        True if the box is valid, False otherwise.
    """
    if isinstance(box, BoundingBox):
        return box.is_valid()
    else:
        bbox = BoundingBox.from_list(box)
        return bbox.is_valid()


def is_box_inside(box1: Union[List[float], BoundingBox], box2: Union[List[float], BoundingBox], threshold: float = 0.0) -> bool:
    """
    Check if box1 is inside box2 (or vice versa) with optional threshold for touching.
    
    Args:
        box1: Bounding box as [x1, y1, x2, y2] or BoundingBox instance
        box2: Bounding box as [x1, y1, x2, y2] or BoundingBox instance
        threshold: Distance threshold for considering boxes as touching
    
    Returns:
        True if one box is inside the other or they're touching
    """
    bbox1 = box1 if isinstance(box1, BoundingBox) else BoundingBox.from_list(box1)
    bbox2 = box2 if isinstance(box2, BoundingBox) else BoundingBox.from_list(box2)
    return bbox1.is_inside(bbox2, threshold)


def are_boxes_touching(box1: Union[List[float], BoundingBox], box2: Union[List[float], BoundingBox], threshold: float = 10.0) -> bool:
    """
    Check if two boxes are touching or overlapping.
    
    Args:
        box1: Bounding box as [x1, y1, x2, y2] or BoundingBox instance
        box2: Bounding box as [x1, y1, x2, y2] or BoundingBox instance
        threshold: Pixel threshold for considering boxes as touching
    
    Returns:
        True if boxes are touching or overlapping
    """
    bbox1 = box1 if isinstance(box1, BoundingBox) else BoundingBox.from_list(box1)
    bbox2 = box2 if isinstance(box2, BoundingBox) else BoundingBox.from_list(box2)
    return bbox1.is_touching(bbox2, threshold)


def merge_boxes(box1: Union[List[float], BoundingBox], box2: Union[List[float], BoundingBox]) -> BoundingBox:
    """
    Merge two boxes into a single bounding box.
    
    Args:
        box1: Bounding box as [x1, y1, x2, y2] or BoundingBox instance
        box2: Bounding box as [x1, y1, x2, y2] or BoundingBox instance
    
    Returns:
        BoundingBox that is the union of both boxes
    """
    bbox1 = box1 if isinstance(box1, BoundingBox) else BoundingBox.from_list(box1)
    bbox2 = box2 if isinstance(box2, BoundingBox) else BoundingBox.from_list(box2)
    return bbox1.merge(bbox2)


def box_contains(box1: Union[List[float], BoundingBox], box2: Union[List[float], BoundingBox], threshold: float = 0.0) -> bool:
    """
    Check if box1 strictly contains box2 (box2 is inside box1).
    
    Args:
        box1: Bounding box as [x1, y1, x2, y2] or BoundingBox instance - potential parent
        box2: Bounding box as [x1, y1, x2, y2] or BoundingBox instance - potential child
        threshold: Distance threshold for growing box1 (box2 remains unchanged)
    
    Returns:
        True if box1 (grown by threshold) contains box2
    """
    bbox1 = box1 if isinstance(box1, BoundingBox) else BoundingBox.from_list(box1)
    bbox2 = box2 if isinstance(box2, BoundingBox) else BoundingBox.from_list(box2)
    return bbox1.contains(bbox2, threshold)


def remove_parent_boxes(boxes: List[Union[List[float], BoundingBox]], threshold: float = 0.0) -> List[BoundingBox]:
    """
    Remove parent bounding boxes from compound speech bubbles and keep children.
    For compound speech bubbles (bubbles within bubbles), removes the parent (outer) 
    box and keeps the child (inner) box.
    
    Args:
        boxes: List of bounding boxes (can be lists or BoundingBox instances)
        threshold: Distance threshold for containment check
    
    Returns:
        List of BoundingBox instances with parent boxes removed
    """
    if not boxes:
        return []
    
    # Convert to BoundingBox instances
    bbox_list = []
    for box in boxes:
        if isinstance(box, BoundingBox):
            bbox_list.append(box)
        else:
            bbox_list.append(BoundingBox.from_list(box))
    
    # Create a copy to work with
    remaining = bbox_list.copy()
    filtered = []
    
    while remaining:
        current_bbox = remaining.pop(0)
        is_parent = False
        
        # Check if current box contains any other box (it's a parent)
        for other_bbox in remaining:
            if current_bbox.contains(other_bbox, threshold=threshold):
                is_parent = True
                break
        
        # Only keep boxes that are not parents
        if not is_parent:
            filtered.append(current_bbox)

    return filtered


def combine_overlapping_bubbles(boxes: List[Union[List[float], BoundingBox]], touch_threshold: float = 10.0) -> List[BoundingBox]:
    """
    Combine compound speech bubble bounding boxes that are inside one another or touching.
    Merges overlapping or adjacent bubbles from compound speech bubbles into single boxes.
    
    Args:
        boxes: List of bounding boxes (can be lists or BoundingBox instances)
        touch_threshold: Pixel threshold for considering boxes as touching
    
    Returns:
        List of merged BoundingBox instances
    """
    if not boxes:
        return []
    
    # Convert to BoundingBox instances
    bbox_list = []
    for box in boxes:
        if isinstance(box, BoundingBox):
            bbox_list.append(box)
        else:
            bbox_list.append(BoundingBox.from_list(box))
    
    # Create a copy to work with
    remaining = bbox_list.copy()
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
            if current_bbox.is_inside(other_bbox, threshold=touch_threshold) or \
               current_bbox.is_touching(other_bbox, threshold=touch_threshold):
                to_merge.append(other_bbox)
                remaining.pop(i)
            else:
                i += 1
        
        # Merge all boxes
        if to_merge:
            # Merge bounding boxes
            for other_bbox in to_merge:
                current_bbox = current_bbox.merge(other_bbox)

            merged.append(current_bbox)
        else:
            # No merging needed, keep as is
            merged.append(current_bbox)

    return merged
