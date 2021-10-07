import os
import shutil


def create_images_folder(root_dir, dest_dir):
    if os.path.exists(dest_dir) and not os.path.isdir(dest_dir):
        print('Error: dest_dir is not a directory')
        return
    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)
    index = 0
    for dir in os.listdir(root_dir):
        for image in os.listdir(os.path.join(root_dir, dir)):
            if image.endswith('.jpg'):
                image_src_path = os.path.join(root_dir, dir, image)
                image_dst_path = os.path.join(dest_dir, str(index) + '.jpg')
                shutil.copyfile(image_src_path, image_dst_path)
                index += 1

if __name__ == '__main__':
    root_dir = os.path.join('ego_hands_data_2', '_LABELLED_SAMPLES')
    dest_dir = os.path.join('ego_hands_data_2', 'all_images')
    create_images_folder(root_dir, dest_dir)