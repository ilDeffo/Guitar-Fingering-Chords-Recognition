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
threshold = 0.799


def get_hand_image_cropped(img, threshold=0.799, padding=100):
    """
    Method to get directly the image cropped after the hands detection.

    :param img: BGR PyTorch image tensor of shape (h, w, c).
    :param threshold: Threshold parameter for hands detection.
    :param padding: Padding value to crop more or less img.
    :return: img cropped around hand if it is detected, otherwise simply img.
    """

    # Directory of model's saved state
    root_dir_saves = os.path.join('hands_detection', 'salvataggi_pytorch', 'trained_two_epochs')

    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(num_classes=2)
    # Loading trained state of the model
    model.load_state_dict(
        torch.load(os.path.join(root_dir_saves, 'model_state_dict.zip'), map_location=torch.device('cpu')))
    model.eval()

    #TODO: Convert image in (c, h, w) tensor with floating point values in range [0, 1] to fit the model's input
    img = img.numpy()
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = torch.from_numpy(img)
    model_input = img.swapaxes(0, 2).swapaxes(1, 2).type(torch.float32) / 255.0
    model_output = model(model_input.unsqueeze(0))
    boxes = model_output[0]['boxes']
    scores = model_output[0]['scores']

    # Taking bounding boxes above threshold.
    boxes, scores = get_boxes_with_score_over_threshold(boxes, scores, threshold)
    if boxes is not None:
        # Taking the rightmost bounding box.
        # It's supposed to be the guitarist's left hand.
        box = get_rightmost_box(boxes)
    else:
        print(f"WARNING! No hand was found in the image {img}. Skipping hands detection...")
        return img

    cropped_image = perform_cropping(img.moveaxis(2, 0), box, padding)
    cropped_image = cropped_image.moveaxis(0, 2).numpy()
    cropped_image = cv.cvtColor(cropped_image, cv.COLOR_RGB2BGR)
    cropped_image = torch.from_numpy(cropped_image)
    return cropped_image

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
