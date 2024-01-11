import xml.etree.cElementTree as ET


def calculate_iou(box1, box2):
    """计算两个边界框的IoU。
    box - [x1, y1, x2, y2]，其中(x1, y1)是左上角坐标，(x2, y2)是右下角坐标。
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - intersection_area

    iou = intersection_area / union_area if union_area != 0 else 0

    return iou


def filter_boxes(boxes, threshold):
    """根据IoU阈值过滤边界框。
    boxes - 边界框列表，每个边界框格式为[x1, y1, x2, y2]。
    threshold - IoU过滤阈值。
    """
    filtered_boxes = []
    for box in boxes:
        keep = True
        for fbox in filtered_boxes:
            if calculate_iou(box, fbox) > threshold:
                keep = False
                break
        if keep:
            filtered_boxes.append(box)

    return filtered_boxes


def create_voc_xml(detections, save_url, width, height, filename, path="", depth=3):
    annotation = ET.Element("annotation")
    ET.SubElement(annotation, "filename").text = filename
    ET.SubElement(annotation, "path").text = path

    source = ET.SubElement(annotation, "source")
    ET.SubElement(source, "database").text = "Unknown"

    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = str(depth)

    for det in detections:
        obj = ET.SubElement(annotation, "object")
        ET.SubElement(obj, "name").text = det['name']
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = str(det['truncated'])
        ET.SubElement(obj, "difficult").text = str(det['difficult'])

        bbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bbox, "xmin").text = str(det['xmin'])
        ET.SubElement(bbox, "ymin").text = str(det['ymin'])
        ET.SubElement(bbox, "xmax").text = str(det['xmax'])
        ET.SubElement(bbox, "ymax").text = str(det['ymax'])

    tree = ET.ElementTree(annotation)
    tree.write(save_url)
