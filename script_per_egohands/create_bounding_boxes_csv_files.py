import os
import scipy.io
import numpy as np
import math
import csv

csv_header = ['image_name', 'box0_x1', 'box0_y1', 'box0_x2', 'box0_y2',  'box1_x1', 'box1_y1', 'box1_x2', 'box1_y2', 'box2_x1', 'box2_y1', 'box2_x2', 'box2_y2', 'box3_x1', 'box3_y1', 'box3_x2', 'box3_y2']

def polygon_to_box(polygon):
    """Convert 1 polygon into a bounding box.
    # Arguments
      polygon: a numpy array of shape (N, 2) representing N vertices
               of the hand segmentation label (polygon); each vertex
               is a point: (x, y)
    """
    if len(polygon) < 3:  # a polygon has at least 3 vertices
        return None

    x_min = np.min(polygon[:, 0])
    y_min = np.min(polygon[:, 1])
    x_max = np.max(polygon[:, 0])
    y_max = np.max(polygon[:, 1])

    x_min = int(math.floor(x_min))
    y_min = int(math.floor(y_min))
    x_max = int(math.ceil(x_max))
    y_max = int(math.ceil(y_max))

    return [x_min, y_min, x_max, y_max]


if __name__ == '__main__':
    index = 0
    for dir in os.listdir(os.path.join('egohands_data', '_LABELLED_SAMPLES')):
        images_dir = os.path.join('egohands_data', '_LABELLED_SAMPLES', dir)
        # carichiamo i poligoni delle immagini in una cartella (es: CARDS_COURTYARD_B_T)
        x = scipy.io.loadmat(os.path.join(images_dir, 'polygons.mat'))
        polygons = x['polygons'][0]

        csvfile = open(os.path.join(images_dir, 'bounding_boxes.csv'), 'w', newline='')
        writer = csv.writer(csvfile)
        writer.writerow(csv_header)

        for arrays in polygons:
            row_to_write = [str(index) + '.jpg']
            index += 1
            # arrays contiene 4 numpy array che contengono i punti che descrivono
            # i poligoni che rappresentano le mani delle persone nelle immagini
            for pol in arrays:
                # otteniamo la bounding box a partire dal poligono
                box = polygon_to_box(pol)
                if box is None:
                    row_to_write += [None, None, None, None]
                else:
                    row_to_write += box

            writer.writerow(row_to_write)
