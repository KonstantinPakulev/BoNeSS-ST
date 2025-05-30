import pandas as pd

from pathlib import Path
from torch.utils.data import Dataset

from source.utils.io import load_image
from source.utils.common_utils import get_offset


class ImageDataset(Dataset):

    def __init__(self, root_path, csv_rel_path, image_transform=None):
        self.root_path = Path(root_path)
        self.df = pd.read_csv(self.root_path / csv_rel_path, index_col=[0])
        self.image_transform = image_transform
    
    def __getitem__(self, index):
        df_row = self.df.iloc[index]
        image_rel_path = Path(df_row['image1'])
        image_path = self.root_path / image_rel_path

        image, grayscale_image = load_image(image_path)

        item = {'index': index,
                'image_filename': image_rel_path.stem,
                'image_dir_rel_path': str(image_rel_path.parent),
                'image': image,
                'grayscale_image': grayscale_image}
        
        if self.image_transform is not None:
            old_image = item['image']
                        
            item['image'] = self.image_transform(old_image)
            item['grayscale_image'] = self.image_transform(item['grayscale_image'])

            item['offset'] = get_offset(old_image, item['image'])
        
        return item
    
    def __len__(self):
        return len(self.df)
