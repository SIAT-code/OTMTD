import torch
from datasets import load_from_disk, dataset_dict
from transformers import ViTFeatureExtractor
from torch.utils.data import Dataset

class CVDataset(Dataset):
    def __init__(self, hf_dataset_dict, set_name='mnist', split='train', return_labels=False, pt_name="google/vit-base-patch16-224"):
        super(CVDataset, self).__init__()
        assert split in ['train', 'test'], 'Only `train` or `test` split, but {} is provided!'.format(split)
        if isinstance(hf_dataset_dict, dataset_dict.DatasetDict):
            pass
        elif isinstance(hf_dataset_dict, str):
            hf_dataset_dict = load_from_disk(hf_dataset_dict)
        else:
            raise ValueError

        self.set_name, self.split, self.return_labels = set_name, split, return_labels
        # Instantiate image preprocessor for VIT
        feature_extractor = ViTFeatureExtractor.from_pretrained(pt_name)

        split_images, split_labels = self.get_split_data(hf_dataset_dict[split])
        self.split_images = feature_extractor(split_images, return_tensors='pt')['pixel_values']
        self.split_labels = torch.tensor(split_labels)

    def __getitem__(self, idx):
        if self.return_labels:
            return self.split_images[idx], self.split_labels[idx]
        else:
            return self.split_images[idx]

    def __len__(self):
        return len(self.split_images)

    def get_split_data(self, hf_dataset):
        if self.set_name in ['mnist', 'fashion_mnist']:
            split_images = hf_dataset['image']
            # convert from L to RGB
            split_images = [img.convert("RGB") for img in split_images]
        else:
            split_images = hf_dataset['img']
        return split_images, hf_dataset['label']
