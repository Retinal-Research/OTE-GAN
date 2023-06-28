from dataloader.EyeQ_Test import EyeQ_Dataset
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

data_transforms = {
        'HQ': T.Compose([
                T.Resize((256,256)),
                #T.RandomRotation((-180,180)),
                T.ToTensor()
        ]),
        'LQ': T.Compose([
                T.Resize((256,256)),
                #T.RandomRotation((-180,180)),
                T.ToTensor()
                #T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

train_set = EyeQ_Dataset(mode='001',transform_HQ=data_transforms['HQ'],transform_PQ=data_transforms['LQ'])
training_data_loader = DataLoader(dataset=train_set,batch_size=1)

print(len(train_set))
for iteration, batch in enumerate(training_data_loader):
       #print(batch)

       print(torch.max(batch[0]))
       print(torch.max(batch[1]))
       print(torch.min(batch[0]))
       print(torch.min(batch[1]))