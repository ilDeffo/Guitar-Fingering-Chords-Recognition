import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision

from guitar_dataset import GuitarDataset

# Nome della cartella in cui salvare le immagini ritagliate
dest_folder = 'cropped_images'

# Scegliamo una soglia. Conserveremo solo le bounding box
# con un punteggio superiore alla soglia
threshold = 0.8

def get_rightmost_box(boxes):
    '''
    Funzione per ottenere la bounding box più a destra nell'immagine
    '''

    values = boxes[:, 2]
    idx = torch.argmax(values)
    return boxes[idx, :]

def perform_cropping(image, box, padding):
    '''
    Funzione che esegue il ritaglio dell'immagine

    :param padding: Spazio aggiuntivo per il ritaglio
    '''

    c, h, w = image.shape
    x1 = box[0] if box[0] - padding < 0 else box[0] - padding
    y1 = box[1] if box[1] - padding < 0 else box[1] - padding
    x2 = box[2] if box[2] + padding > w else box[2] + padding
    y2 = box[3] if box[3] + padding > h else box[3] + padding

    x1 = int(torch.floor(x1))
    y1 = int(torch.floor(y1))
    x2 = int(torch.floor(x2))
    y2 = int(torch.floor(y2))

    return image[:, y1:y2, x1:x2]

def get_label_name(label):
    '''
    Funzione per ottenere il nome della classe
    a partire dal suo codice numerico
    '''

    from guitar_dataset import label_mappings
    for k, v in label_mappings.items():
        if v == label:
            return k

def save_image(idx, image, label, dest_folder):
    '''
    Funzione per salvare un'immagine in una cartella
    '''

    image = np.moveaxis(image.numpy(), 0, 2)
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    image *= 255
    image = image.round().clip(0, 255)
    label_name = get_label_name(label)
    im_name = label_name + ' (' + str(idx) + ')' + '.jpeg'
    out_path = os.path.join(dest_folder, im_name)
    cv.imwrite(out_path, image)

def get_boxes_with_score_over_threshold(boxes, scores, threshold):
    '''
    Funzione per ottenere le bounding box che hanno un punteggio maggiore della soglia

    :return: Una tupla (boxes, scores)
    '''
    final_boxes = None
    for box, score in zip(boxes, scores):
        if score > threshold:
            if final_boxes is None:
                final_boxes = torch.clone(box).reshape(1, 4)
            else:
                final_boxes = torch.cat((final_boxes, box.reshape(1, 4)))

    if final_boxes is not None:
        return final_boxes, scores[:final_boxes.shape[0]]
    else:
        return None, None


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

    for idx, (image, label) in enumerate(guitar_dataset):
        out = model(image.unsqueeze(0))
        #print(out)
        boxes = out[0]['boxes']
        scores = out[0]['scores']

        # Prendiamo le bounding box che hanno un punteggio superiore ad una soglia.
        boxes, scores = get_boxes_with_score_over_threshold(boxes, scores, threshold)

        if boxes.shape[0] > 0:
            # Prendiamo la bounding box più a destra nell'immagine.
            # Molto probabilmente sarà la bounding box della mano sinistra del chitarrista.
            box = get_rightmost_box(boxes)
        else:
            continue

        padding = 100
        new_image = perform_cropping(image, box, padding)
        save_image(idx, new_image, label, dest_folder)
