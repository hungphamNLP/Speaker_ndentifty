import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import torchvision.utils
import numpy as np
import random
from PIL import Image
import torch
import PIL.ImageOps    
import torch.nn.functional as F
from utils import Config,imshow

class SiameseNetworkDataset(Dataset):
    
    def __init__(self,imageFolderDataset,transform=None,should_invert=True):
        self.imageFolderDataset = imageFolderDataset    
        self.transform = transform
        self.should_invert = should_invert
        
    def __getitem__(self,index):
        img0_tuple = random.choice(self.imageFolderDataset.imgs)
        #we need to make sure approx 50% of images are in the same class
        should_get_same_class = random.randint(0,1) 
        if should_get_same_class:
            while True:
                img1_tuple = random.choice(self.imageFolderDataset.imgs) 
                if img0_tuple[1]==img1_tuple[1]:
                    break
        else:
            while True:
                img1_tuple = random.choice(self.imageFolderDataset.imgs) 
                if img0_tuple[1] !=img1_tuple[1]:
                    break

        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])

        ## Conversion to grayscale
        img0 = img0.convert("L")
        img1 = img1.convert("L")
        
        if self.should_invert:
            img0 = PIL.ImageOps.invert(img0)
            img1 = PIL.ImageOps.invert(img1)

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
        labels = torch.squeeze(torch.from_numpy(np.array([int(img1_tuple[1]==img0_tuple[1])],dtype=np.float32)).long())
        ## Set the  Label to 1 when the images are from the same class
        print(img0.shape)
        return img0, img1 , labels
    
    def __len__(self):
        return len(self.imageFolderDataset.imgs)



use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print('Using device', device)
import multiprocessing
num_workers = multiprocessing.cpu_count()
print('num workers:', num_workers)
kwargs = {'num_workers': num_workers,
        'pin_memory': True} if use_cuda else {}
folder_dataset = dset.ImageFolder(root=Config.training_dir)
siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset,
                                    transform=transforms.Compose([#transforms.Resize((227,227)),
                                                                    transforms.ToTensor()
                                                                    ])
                                    ,should_invert=False)
train_dataloader = DataLoader(siamese_dataset,
                            shuffle=True,
                            batch_size=Config.train_batch_size,
                            **kwargs)



folder_dataset_test = dset.ImageFolder(root=Config.testing_dir)
siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset_test,
                                        transform=transforms.Compose([transforms.ToTensor()]),
                                        should_invert=False)

test_dataloader = DataLoader(siamese_dataset, batch_size=2, shuffle=False, **kwargs)



if __name__ == '__main__':

    
    vis_dataloader = DataLoader(siamese_dataset, shuffle=True,         
                            batch_size=2, **kwargs)
    dataiter = iter(vis_dataloader)


    example_batch = next(dataiter)
    concatenated = torch.cat((example_batch[0],example_batch[1]),0)
    imshow(torchvision.utils.make_grid(concatenated))
    print(example_batch[2].numpy())