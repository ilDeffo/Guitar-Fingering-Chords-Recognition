'''
Questo script ha lo stesso funzionamento dello script "detect_hands.py".
Tuttavia, invece di iterare su tutto il nostro dataset, indichiamo un'immagine
specifica. Infatti può accadere che la soglia impostata nel file "detect_hands.py"
non sia ottimale per tutte le immagini. Dunque è possibile provare con un valore
diverso della soglia.
'''

import os
import matplotlib.pyplot as plt
import torch
import torchvision
import cv2 as cv

from guitar_dataset import GuitarDataset

from detect_hands import get_rightmost_box
from detect_hands import perform_cropping
from detect_hands import save_image
from detect_hands import get_boxes_with_score_over_threshold

# Nome della cartella in cui salvare l'immagine ritagliata
dest_folder = 'cropped_images'

# Scegliamo una soglia. Conserveremo solo le bounding box
# con un punteggio superiore alla soglia
threshold = 0.79

# Temporary directory to save bounding boxes drawed on image
TMP_DIR = "Temp" + os.sep


def detect_hand(img, threshold=0.79, save_img=True, verbose=False):
    """
    Method to detect the guitarist's hand playing the chord using the FasterCNN network.

    :param img: BGR PyTorch image tensor of shape (h, w, c).
    :param threshold: Threshold parameter for hands detection.
    :param verbose:
    :param save_img: If true, the algorithm saves image result in TMP_DIR.
    :return: Dictionary with bounding box coordinates and score of detection
             if the hand is found, otherwise None.
    """

    # Directory of model's saved state
    root_dir_saves = os.path.join('hands_detection', 'salvataggi_pytorch', 'trained_two_epochs')

    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(num_classes=2)
    # Loading trained state of the model
    model.load_state_dict(
        torch.load(os.path.join(root_dir_saves, 'model_state_dict.zip'), map_location=torch.device('cpu')))
    model.eval()

    # Converting image in (c, h, w) RGB Numpy array and then back to Torch tensor with floating point
    # values in range [0, 1] to fit the model's input
    img = cv.cvtColor(img.numpy(), cv.COLOR_BGR2RGB)
    img = torch.from_numpy(img)
    model_input = img.swapaxes(0, 2).swapaxes(1, 2).type(torch.float32) / 255.0

    model_output = model(model_input.unsqueeze(0))
    boxes = model_output[0]['boxes']
    scores = model_output[0]['scores']

    # Taking bounding boxes above threshold and initalizing box and score in output.
    boxes, scores = get_boxes_with_score_over_threshold(boxes, scores, threshold, verbose)
    box, score = None, None

    if boxes is not None:
        # Taking the rightmost bounding box, supposed to be the guitarist's left hand.
        box = get_rightmost_box(boxes, scores, verbose=verbose)

        for idx, b in enumerate(boxes):
            if torch.equal(b, box):
                # Taking score of the rightest box found
                score = scores[idx].item()

            # Drawing bounding box on original image
            b = b.detach().numpy().astype(int)
            cv.rectangle(img.numpy(), (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)

        if save_img:
            # Saving drawed image in temporary directory
            img = cv.cvtColor(img.numpy(), cv.COLOR_RGB2BGR)
            cv.imwrite(TMP_DIR + 'hands_detection.jpg', img)

    return {'box': box, 'score': score}


def get_hand_image_cropped(img, threshold=0.799, padding=100, verbose=False, save_img=True):
    """
    Method to get directly the image cropped after the hands detection.
    It calls the method detect_hand, so be aware that an image with bounding boxes
    drawed will be saved in the TMP_DIR specified in the script.

    :param img: BGR PyTorch image tensor of shape (h, w, c).
    :param threshold: Threshold parameter for hands detection.
    :param padding: Padding value to crop more or less img.
    :param verbose:
    :param save_img: If true, the algorithm saves image result in TMP_DIR.
    :return: img cropped around hand if it is detected, otherwise simply img.
    """
    # Getting bounding box and score
    detection = detect_hand(img, threshold, save_img, verbose)
    box, score = detection['box'], detection['score']

    if box is None or score is None:
        print(f"WARNING! No hand was found in the image {img.shape}. Skipping hands detection...")
        return img

    if verbose:
        print(f"Hand found in box {box} with score {score}! Cropping image around it...")

    cropped_image = perform_cropping(img.moveaxis(2, 0), box, padding)
    out = cropped_image.moveaxis(0, 2)
    """
    The following lines are needed only if image has been already converted before to fit the model's input.
    
    # Returning as out the cropped image converted to Torch BGR tensor of shape (h, w, c)
    out = cv.cvtColor(out.numpy(), cv.COLOR_RGB2BGR)
    out = torch.from_numpy(out)
    """
    return out


if __name__ == '__main__':
    # Percorso del nostro dataset
    root_dir_guitar_dataset = os.path.join('Dataset', 'all_images')

    guitar_dataset = GuitarDataset(root_dir_guitar_dataset)

    # Percorso in cui sono contenuti i salvataggi dello stato della rete neurale
    root_dir_saves = os.path.join('hands_detection', 'salvataggi_pytorch', 'trained_two_epochs')

    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(num_classes=2)
    # Carichiamo lo stato salvato della rete neurale
    model.load_state_dict(torch.load(os.path.join(root_dir_saves, 'model_state_dict.zip'), map_location=torch.device('cpu')))
    model.eval()

    image_index = 328

    image, label = guitar_dataset[image_index]
    out = model(image.unsqueeze(0))
    # print(out)
    boxes = out[0]['boxes']
    scores = out[0]['scores']

    # Prendiamo le bounding box che hanno un punteggio superiore ad una soglia.
    boxes, scores = get_boxes_with_score_over_threshold(boxes, scores, threshold)
    if boxes.shape[0] > 0:
        # Prendiamo la bounding box più a destra nell'immagine.
        # Molto probabilmente sarà la bounding box della mano sinistra del chitarrista.
        box = get_rightmost_box(boxes)
    else:
        import sys
        sys.exit(0)

    padding = 100
    new_image = perform_cropping(image, box, padding)
    save_image(image_index, new_image, label, dest_folder)
