import json
import os
import pandas as pd

from torch.utils.data import Dataset


class TransformedDataset(Dataset):
    def __init__(self, dataset, builder, local_image_dir='./data/coco', data_format='hf'):
        super(TransformedDataset, self).__init__()
        self.builder = builder
        self.local_image_dir = local_image_dir
        self.data_format = data_format
        self.raw_data = []
        self.raw_data = self.load_raw_data(dataset)

    def load_raw_data(self, dataset):
        if self.data_format == 'hf':
            return self.load_hf_data(dataset)
        elif self.data_format == 'parquet':
            return self.load_parquet_data(dataset)
        else:
            raise ValueError(f"Invalid data_format: {self.data_format}")
        
    def load_hf_data(self, dataset):
        # loading data from Huggingface dataset
        dataset = dataset['train']
        image_names = dataset['image']
        for index, img_name in enumerate(image_names):  
            local_path = os.path.join(self.local_image_dir, img_name)
            
            if os.path.exists(local_path):
                with open(local_path, 'rb') as f:
                    buffer = f.read()
            else:
                raise ValueError(f"Image file not found: {local_path}")
            
            DATA = {}
            DATA['BUFFER'] = buffer
            conversation_data = dataset['conversations'][index]
            DATA['ZH_TEXT'] = json.dumps(conversation_data, ensure_ascii=False)
            self.raw_data.append(DATA)
        return self.raw_data
    
    def __len__(self):
        return len(self.raw_data)
    
    def __getitem__(self, idx):
        data = self.raw_data[idx]
        item = self.builder.build_item(data)
        return item[0]