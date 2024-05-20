import os
from glob import glob
import numpy as np
from PIL import Image

BG20_DIR = "../dbs/BG20K"
# used for generating background candidates
class BG20K4LABEL():
    def __init__(self, split, dbdir=BG20_DIR):
        self.filenames = glob(os.path.join(dbdir, split, "*.jpg"))
        assert len(self.filenames) > 0, "No folders found, please check database dir"
        np.sort(self.filenames, axis=0)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = self.filenames[idx]
        image = Image.open(img_path)
        self.sample = {"img_data": image, "path": os.path.basename(img_path), 'img_path': img_path}
        return self.sample


def get_id_name_dict(output_path='./metadata/filename_dict.txt'):
    with open(output_path, 'w') as f:
        for split in ['val', 'train']:
            db = BG20K4LABEL(split)
            for i in range(len(db)):
                filename = os.path.basename(db[i]['path']).split('.')[0]
                f.write(f'{i} {split} {filename}\n')

if __name__ == '__main__':
    get_id_name_dict()

