import os
import torch
import torch.utils.data
from PIL import Image
from datasets.data_augment import PairCompose, PairToTensor, PairRandomHorizontalFilp
from torchvision import transforms

class LLdataset:
    def __init__(self, config):
        self.config = config

    def get_loaders(self):
        train_dataset = AllWeatherDataset(self.config.data.data_dir,
                                          patch_size=self.config.data.patch_size,
                                          filelist='{}_train.txt'.format(self.config.data.train_dataset))
        val_dataset = AllWeatherDataset(self.config.data.data_dir,
                                        patch_size=self.config.data.patch_size,
                                        filelist='{}_val.txt'.format(self.config.data.val_dataset), train=False)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.config.training.batch_size,
                                                   shuffle=True, num_workers=self.config.data.num_workers,
                                                   pin_memory=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.config.sampling.batch_size,
                                                 shuffle=False, num_workers=self.config.data.num_workers,
                                                 pin_memory=True)

        return train_loader, val_loader


class AllWeatherDataset(torch.utils.data.Dataset):
    def __init__(self, dir, patch_size, filelist=None, train=True):
        super().__init__()

        self.dir = dir
        self.file_list = filelist
        self.train_list = os.path.join(dir, self.file_list)
        with open(self.train_list) as f:
            contents = f.readlines()
            input_names = [i.strip() for i in contents]

        self.input_names = input_names
        self.patch_size = patch_size
        self.resize = transforms.Resize((384, 512))

        
        if train:
            self.transforms = PairCompose([
                PairRandomHorizontalFilp(),
                PairToTensor()
            ])
        else:
            self.transforms = PairCompose([
                PairToTensor()
            ])
        

    def get_images(self, index):
        input_name = self.input_names[index].replace('\n', '')

        #print(f"DEBUG: input_name = '{input_name}'")  # 🔍 데이터 확인용 추가

        split_name = input_name.split(' ')
        if len(split_name) < 2:
            raise ValueError(f"Error: input_name does not contain two file paths! input_name = {input_name}")

        low_img_name, high_img_name = split_name[0], split_name[1]
        low_img, high_img = Image.open(low_img_name), Image.open(high_img_name)

        # 이미지 크기 조정
        low_img, high_img = self.resize(low_img), self.resize(high_img)
        low_img, high_img = self.transforms(low_img, high_img)

        return torch.cat([low_img, high_img], dim=0), input_name

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.input_names)
