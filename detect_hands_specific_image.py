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
