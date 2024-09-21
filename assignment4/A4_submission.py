import numpy as np
import torch
import os
import shutil
import torchvision
from torchvision.ops import masks_to_boxes
from Unet import UNET
from FasterRcnn_model import *
import pathlib

def generate_pred_bboxes(result):
    # print(result)
    boxes = result[0]['boxes'].detach().round().cpu().numpy()
    if len(boxes) >= 2:
        pred_box = np.empty((2, 4), dtype=np.float64)
        labels = result[0]['labels'].detach().cpu().numpy().astype(np.int32)
        if labels[0] < labels[1]:
            pred_box[0] = np.array([boxes[0][1], boxes[0][0], boxes[0][3], boxes[0][2]])
            pred_box[1] = np.array([boxes[1][1], boxes[1][0], boxes[1][3], boxes[1][2]])
            pred_clas = np.array([labels[0]-1, labels[1]-1], dtype=np.int32)
        else:
            pred_box[0] = np.array([boxes[1][1], boxes[1][0], boxes[1][3], boxes[1][2]])
            pred_box[1] = np.array([boxes[0][1], boxes[0][0], boxes[0][3], boxes[0][2]])
            pred_clas = np.array([labels[1]-1, labels[0]-1], dtype=np.int32)
    else:
        boxes = result[0]['boxes'].detach().round().cpu().numpy()
        labels = result[0]['labels'].detach().cpu().numpy().astype(np.int32)
        if len(boxes) > 0:
            pred_box = np.empty((2, 4), dtype=np.float64)
            pred_box[0] = np.array([boxes[0][1], boxes[0][0], boxes[0][3], boxes[0][2]])
            pred_box[1] = np.array([boxes[0][1], boxes[0][0], boxes[0][3], boxes[0][2]])
            pred_clas = np.array([0, labels[0]-1], dtype=np.int32)

        else:
            pred_box = np.empty((2, 4), dtype=np.float64)
            pred_clas = np.empty((2), dtype=np.int32)

    return pred_box, pred_clas.reshape(2)

def detect_and_segment(images):
    """

    :param np.ndarray images: N x 12288 array containing N 64x64x3 images flattened into vectors
    :return: np.ndarray, np.ndarray
    """
    N = images.shape[0]

    # pred_class: Your predicted labels for the 2 digits, shape [N, 2]
    pred_class = np.empty((N, 2), dtype=np.int32)
    # pred_bboxes: Your predicted bboxes for 2 digits, shape [N, 2, 4]
    pred_bboxes = np.empty((N, 2, 4), dtype=np.float64)
    # pred_seg: Your predicted segmentation for the image, shape [N, 4096]
    pred_seg = np.empty((N, 4096), dtype=np.int32)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    path_dir = pathlib.Path(__file__).parent.resolve()
    unet_MODEL_PATH = os.path.join(path_dir, "saved_model.pth")
    faster_rcnn_MODEL_PATH = os.path.join(path_dir, 'object_detection_rcnn1.pth')

    # unet model
    unet_model = UNET(3, 11).to(device)
    unet_model.eval()
    checkpoint = torch.load(unet_MODEL_PATH)
    unet_model.load_state_dict(checkpoint["model_state_dict"])

    # faster rcnn model
    faster_rcnn_model = create_model(num_classes=11, pretrained=True, coco_model=False).to(device)
    faster_rcnn_model.eval()
    checkpoint2 = torch.load(faster_rcnn_MODEL_PATH)
    faster_rcnn_model.load_state_dict(checkpoint2['model_state_dict'])

    transforms_image = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    images = images.reshape(N, 64, 64, 3)

    for i in range(N):
        # print(images[i].shape)
        image = transforms_image(images[i])
        # print(image.shape)
        image2 = [image.clone().to(device)]
        # print(image2)
        image = image.reshape(1, 3, 64, 64)
        image = image.to(device)
        pred_seg[i] = unet_model(image).argmax(dim=1).flatten().cpu()

        output = faster_rcnn_model(image2)
        pred_bboxes[i], pred_class[i] = generate_pred_bboxes(output)

    return pred_class, pred_bboxes, pred_seg