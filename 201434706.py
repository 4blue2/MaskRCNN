import torch
import cv2
import torchvision
import numpy as np
import random
from math import atan2, cos, sin, sqrt, pi

# Functions get_orientation() and draw_axis obtained from https://docs.opencv.org/4.5.3/d1/dee/tutorial_introduction_to_pca.html

def get_orientation(pts, img):
    sz = len(pts)
    data_pts = np.empty((sz, 2), dtype=np.float64)
    for i in range(data_pts.shape[0]):
        data_pts[i,0] = pts[i,0,0]
        data_pts[i,1] = pts[i,0,1]
    # Perform PCA analysis
    mean = np.empty((0))
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)
    # Store the center of the object
    cntr = (int(mean[0,0]), int(mean[0,1]))
    cv2.circle(img, cntr, 3, (255, 0, 255), 2)
    p1 = (cntr[0] + 0.02 * eigenvectors[0,0] * eigenvalues[0,0], cntr[1] + 0.02 *  eigenvectors[0,1] * eigenvalues[0,0])
    p2 = (cntr[0] - 0.02 * eigenvectors[1,0] * eigenvalues[1,0], cntr[1] - 0.02 * eigenvectors[1,1] * eigenvalues[1,0])
    draw_axis(img, cntr, p1, (0, 150, 0), 1)
    draw_axis(img, cntr, p2, (200, 150, 0), 2)
    angle = atan2(eigenvectors[0,1], eigenvectors[0,0]) # orientation in radians
    return angle

def draw_axis(img, p_, q_, colour, scale):
    p = list(p_)
    q = list(q_)
    angle = atan2(p[1] - q[1], p[0] - q[0]) # angle in radians
    hypotenuse = sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0]))
    # Here we lengthen the arrow by a factor of scale
    q[0] = p[0] - scale * hypotenuse * cos(angle)
    q[1] = p[1] - scale * hypotenuse * sin(angle)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)
    # create the arrow hooks
    p[0] = q[0] + 9 * cos(angle + pi / 4)
    p[1] = q[1] + 9 * sin(angle + pi / 4)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)
    p[0] = q[0] + 9 * cos(angle - pi / 4)
    p[1] = q[1] + 9 * sin(angle - pi / 4)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)



# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device -> ', device)

# Load the pre-trained Mask R-CNN model from torchvision
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model.to(device)
model.eval()


classes = ['unlabeled', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'street sign', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'hat', 'backpack', 'umbrella', 'shoe', 'eye glasses', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'plate', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'mirror', 'dining table', 'window', 'desk', 'toilet', 'door', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'blender', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
colors = [[random.randint(0, 255) for _ in range(3)] for _ in classes]


# Init. the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam,
    # returns two values, we are only interested in the frame
    _, frame = cap.read()

    # Convert the frame to a PyTorch tensor
    input_tensor = torchvision.transforms.functional.to_tensor(frame)

    # Run the input tensor through the model
    output = model([input_tensor])      

    # Get the predicted boxes, labels, and masks
    boxes = output[0]["boxes"].detach().numpy()
    labels = output[0]["labels"].detach().numpy()
    scores = output[0]["scores"].detach().numpy()
    masks = output[0]["masks"].detach().numpy()

    # Iterate over the predicted objects and draw bounding boxes and masks
    for i in range(len(boxes)):
        if scores[i] > 0.5:
            # Get the label and color for this object
            label = classes[labels[i]]
            color = random.choice(colors)

            # Draw the bounding box
            x1, y1, x2, y2 = boxes[i]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw the mask
            mask = masks[i, 0]
            mask = cv2.resize(mask, (x2 - x1, y2 - y1))
            mask = (mask > 0.5).astype(np.uint8) #convert it to binary mask with values above 0.5 = 1
            mask_color = np.zeros((y2 - y1, x2 - x1, 3), dtype=np.uint8)
            mask_color[:, :, 0] = color[0] #give color to the mask

            # Make changes inside the box boundaries,
            # Replace frame with mask color at pixels where mask is = 1
            frame[y1:y2, x1:x2] = np.where(mask[..., np.newaxis] > 0, mask_color, frame[y1:y2, x1:x2])

            #gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            _, bw = cv2.threshold(mask, 50, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

            contours, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

            for i,c in enumerate(contours):
                # area of each contour
                area = cv2.contourArea(c)
                # ignore contour which is too small or large
                if area < 1e2 or 1e5 < area:
                    continue
                # draw each contour only for visualization
                cv2.drawContours(frame, contours, i, (23, 120, 156), 2)
                # find orientation of each shape
                get_orientation(c,frame)

            # Draw the label and score
            label_text = f"{label}: {int(scores[i] * 100)}%"
            cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # Will be used for calculating the center point
            num_rows, num_col = mask.shape

            # num_rows/2, num_col/2 get center point of the mask tensor
            # then take init points of box and add center points of mask to place circle in the center of mask
            cv2.circle(frame, (int(x1+(num_rows/2)), int(y1+(num_col/2))), radius=3, color=(0,0,255), thickness=4)

    # Show the resulting frame
    cv2.imshow("Mask R-CNN", frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
