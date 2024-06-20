import torch
import numpy as np
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import pickle
from torch.utils import data
from PIL import Image

data_stats=pickle.load(open('./userdataset.pkl','rb'))


def load_dataset(user,args):
    data_path = './users_data/' + user +'/'
    
    full_dataset = torchvision.datasets.ImageFolder(root=data_path,transform = transforms.Compose(
        [torchvision.transforms.Resize((32,32)),
         torchvision.transforms.ToTensor()]))                                                                                                    
    train_size = int(args.split_ratio * len(full_dataset))
    test_size = len(full_dataset) - train_size
    torch.manual_seed(178)
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.bs, num_workers=2, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.bs, num_workers=2, shuffle=False)
    
    return train_size,train_loader,test_loader

def new_load_dataset(user,args):
    train_dataset = userdata_loader('train',user,args)
    test_dataset = userdata_loader('test',user,args)
    # print(len(train_dataset),"Length of train data")
    # print(len(test_dataset),"Length of test data")
    train_loader = data.DataLoader(train_dataset, batch_size=args.bs, num_workers=2, shuffle=True)
    test_loader = data.DataLoader(test_dataset, batch_size=args.bs, num_workers=2, shuffle=False)
    return len(train_dataset) , train_loader, test_loader

class userdata_loader(data.Dataset):
    # adj_matrix=pickle.load(open(graphfile,'rb'))

    def __init__(self,mode = None, user = None, args=None):
        self.dataset = data_stats[user]
        self.mode = mode
        self.args=args
        self.ids = [i for i in range(len(self.dataset))]
        self.get_split()
        # self.dataset=pickle.load(open(datafile,'rb'))
        # print(len(self.dataset))
        # print(len(self.sampleids))
        # self.imgbase=imgbase
        self.preprocessing=transforms.Compose([
                                        transforms.Resize((32,32)),
                                        transforms.ToTensor()
                                        # transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
                                        ])

    def get_split(self,):
        trainsize = int(self.args.split_ratio * len(self.dataset))
        # trainsize = int(0.8 * trainfullsize)
        np.random.seed(self.args.seed)
        np.random.shuffle(self.ids)
        if self.mode == 'train':
            self.sampleids = self.ids[:trainsize]
        else:
            self.sampleids = self.ids[trainsize:]

    def get_image(self,imgpath):
        # print(img_no)
        # imgpath=os.path.join(imgbase,img_no+'.jpg')
        # print("In Get Image ", imgpath)
        img = Image.open(imgpath)
        # if img.mode == 'RGBA':
        #     img=self.rgb(img)
        # w,h=img.size
        img=img.convert('RGB')
        # print(img.size,"Image Size")
        # pad_value=(int(max(h-w,0)/2),int(max(w-h,0)/2),int(max(h-w,0)/2),int(max(w-h,0)/2))
        # img = TF.pad(img,pad_value)
        return self.preprocessing(img)

    def __getitem__(self,index):
        # print(self.dataset[self.sampleids[index]])
        # label=torch.tensor([self.dataset[self.sampleids[index]]['label']])
        label=self.dataset[self.sampleids[index]]['label']
        weight=1.0/self.dataset[self.sampleids[index]]['weight'] if self.args.naive else 1.0
        weight=torch.tensor([weight])
        # print(weight,"Weight Value")
        img=self.get_image(self.dataset[self.sampleids[index]]['path']) 
        # print(img.shape,"Image")
        # print(label.shape,"Label")
        # print(weight.shape,"weight")
        return img,label,weight


    def __len__(self):
        return len(self.sampleids)