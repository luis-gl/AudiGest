import os
import shutil


def remove_elements(parent_dir: str):
    for content in os.listdir(parent_dir):
        content_path = os.path.join(parent_dir, content)
        if not content.startswith('level'):
            remove_elements(content_path)
            continue

        lmks_path = os.path.join(content_path, 'landmarks')
        lmks = os.listdir(lmks_path)
        for lmk in lmks:
            lmk_element = os.path.join(lmks_path, lmk)
            if os.path.exists(lmk_element) and os.path.isdir(lmk_element):
                shutil.rmtree(lmk_element)

        mfcc_path = os.path.join(content_path, 'mfcc')
        if os.path.exists(mfcc_path) and os.path.isdir(mfcc_path):
            shutil.rmtree(mfcc_path)


def main():
    root = 'processed_data'
    phase = 'test'

    remove_elements(os.path.join(root, phase))


if __name__ == '__main__':
    main()
