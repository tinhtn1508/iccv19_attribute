import os
import sys
from PIL import Image
import torch
import numpy as np
from torch._C import dtype
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import pickle

def default_loader(path):
    return Image.open(path).convert('RGB')
# class MultiLabelDataset(data.Dataset):
#     def __init__(self, root, label, transform = None, loader = default_loader):
#         images = []
#         labels = open(label).readlines()
#         for line in labels:
#             items = line.split()
#             img_name = items.pop(0)
#             if os.path.isfile(os.path.join(root, img_name)):
#                 cur_label = tuple([int(v) for v in items])
#                 images.append((img_name, cur_label))
#             else:
#                 print(os.path.join(root, img_name) + 'Not Found.')
#         self.root = root
#         self.images = images
#         self.transform = transform
#         self.loader = loader

#     def __getitem__(self, index):
#         img_name, label = self.images[index]
#         img = self.loader(os.path.join(self.root, img_name))
#         raw_img = img.copy()
#         if self.transform is not None:
#             img = self.transform(img)
#         return img, torch.Tensor(label)

#     def __len__(self):
#         return len(self.images)

class MultiLabelDataset(data.Dataset):
    def __init__(self, split, data_path, transform = None, loader = default_loader):
        dataset_info = pickle.load(open(data_path, 'rb+'))
        img_id = dataset_info.image_name
        attr_label = dataset_info.label

        self.transform = transform
        self.loader = loader
        self.root_path = dataset_info.root
        self.attr_id = dataset_info.attr_name
        self.attr_num = len(self.attr_id)
        self.img_idx = dataset_info.partition[split]

        self.img_num = self.img_idx.shape[0]
        self.img_id = [img_id[i] for i in self.img_idx]
        self.label = attr_label[self.img_idx]


    def __getitem__(self, index):
        imgname, gt_label, imgidx = self.img_id[index], self.label[index], self.img_idx[index]
        imgpath = os.path.join(self.root_path, imgname)
        img = self.loader(imgpath)
        
        if self.transform is not None:
            img = self.transform(img)
        return img, torch.Tensor(gt_label)

    def __len__(self):
        return len(self.img_id)

attr_nums = {}
attr_nums['pa100k'] = 26
attr_nums['rap'] = 51
attr_nums['peta'] = 105

description = {}
description['pa100k'] = ['Female',
                        'AgeOver60',
                        'Age18-60',
                        'AgeLess18',
                        'Front',
                        'Side',
                        'Back',
                        'Hat',
                        'Glasses',
                        'HandBag',
                        'ShoulderBag',
                        'Backpack',
                        'HoldObjectsInFront',
                        'ShortSleeve',
                        'LongSleeve',
                        'UpperStride',
                        'UpperLogo',
                        'UpperPlaid',
                        'UpperSplice',
                        'LowerStripe',
                        'LowerPattern',
                        'LongCoat',
                        'Trousers',
                        'Shorts',
                        'Skirt&Dress',
                        'boots']

description['peta'] = ['personalLess30',
                        'personalLess45',
                        'personalLess60',
                        'personalLarger60',
                        'carryingBackpack',
                        'carryingOther',
                        'lowerBodyCasual',
                        'upperBodyCasual',
                        'lowerBodyFormal',
                        'upperBodyFormal', 'accessoryHat', 'upperBodyJacket', 'lowerBodyJeans', 'footwearLeatherShoes', 'upperBodyLogo', 'hairLong', 'personalMale', 'carryingMessengerBag', 'accessoryMuffler', 'accessoryNothing', 'carryingNothing', 'upperBodyPlaid', 'carryingPlasticBags', 'footwearSandals', 'footwearShoes', 'lowerBodyShorts', 'upperBodyShortSleeve', 'lowerBodyShortSkirt', 'footwearSneaker', 'upperBodyThinStripes', 'accessorySunglasses', 'lowerBodyTrousers', 'upperBodyTshirt', 'upperBodyOther', 'upperBodyVNeck', 'upperBodyBlack', 'upperBodyBlue', 'upperBodyBrown', 'upperBodyGreen', 'upperBodyGrey', 'upperBodyOrange', 'upperBodyPink', 'upperBodyPurple', 'upperBodyRed', 'upperBodyWhite', 'upperBodyYellow', 'lowerBodyBlack', 'lowerBodyBlue', 'lowerBodyBrown', 'lowerBodyGreen', 'lowerBodyGrey', 'lowerBodyOrange', 'lowerBodyPink', 'lowerBodyPurple', 'lowerBodyRed', 'lowerBodyWhite', 'lowerBodyYellow', 'hairBlack', 'hairBlue', 'hairBrown', 'hairGreen', 'hairGrey', 'hairOrange', 'hairPink', 'hairPurple', 'hairRed', 'hairWhite', 'hairYellow', 'footwearBlack', 'footwearBlue', 'footwearBrown', 'footwearGreen', 'footwearGrey', 'footwearOrange', 'footwearPink', 'footwearPurple', 'footwearRed', 'footwearWhite', 'footwearYellow', 'accessoryHeadphone', 'personalLess15', 'carryingBabyBuggy', 'hairBald', 'footwearBoots', 'lowerBodyCapri', 'carryingShoppingTro', 'carryingUmbrella', 'personalFemale', 'carryingFolder', 'accessoryHairBand', 'lowerBodyHotPants', 'accessoryKerchief', 'lowerBodyLongSkirt', 'upperBodyLongSleeve', 'lowerBodyPlaid', 'lowerBodyThinStripes', 'carryingLuggageCase', 'upperBodyNoSleeve', 'hairShort', 'footwearStocking', 'upperBodySuit', 'carryingSuitcase', 'lowerBodySuits', 'upperBodySweater', 'upperBodyThickStripes']

description['rap'] = ['Female',
                        'AgeLess16',
                        'Age17-30',
                        'Age31-45',
                        'BodyFat',
                        'BodyNormal',
                        'BodyThin',
                        'Customer',
                        'Clerk',
                        'BaldHead',
                        'LongHair',
                        'BlackHair',
                        'Hat',
                        'Glasses',
                        'Muffler',
                        'Shirt',
                        'Sweater',
                        'Vest',
                        'TShirt',
                        'Cotton',
                        'Jacket',
                        'Suit-Up',
                        'Tight',
                        'ShortSleeve',
                        'LongTrousers',
                        'Skirt',
                        'ShortSkirt',
                        'Dress',
                        'Jeans',
                        'TightTrousers',
                        'LeatherShoes',
                        'SportShoes',
                        'Boots',
                        'ClothShoes',
                        'CasualShoes',
                        'Backpack',
                        'SSBag',
                        'HandBag',
                        'Box',
                        'PlasticBag',
                        'PaperBag',
                        'HandTrunk',
                        'OtherAttchment',
                        'Calling',
                        'Talking',
                        'Gathering',
                        'Holding',
                        'Pusing',
                        'Pulling',
                        'CarryingbyArm',
                        'CarryingbyHand']




def Get_Dataset(experiment, approach):

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform_train = transforms.Compose([
        transforms.Resize(size=(256, 128)),
        transforms.RandomHorizontalFlip(),
        # transforms.ColorJitter(hue=.05, saturation=.05),
        # transforms.RandomRotation(20, resample=Image.BILINEAR),
        transforms.ToTensor(),
        normalize
        ])
    # transform_test = transforms.Compose([
    #     transforms.Resize(size=(256, 128)),
    #     transforms.ToTensor(),
    #     normalize
    #     ])

    if experiment == 'pa100k':
        train_dataset = MultiLabelDataset(split="train",
                    data_path='./dataset/dataset.pkl', transform=transform_train)
        val_dataset = MultiLabelDataset(split="test",
                    data_path='./dataset/dataset.pkl', transform=transform_train)
        return train_dataset, val_dataset, attr_nums['pa100k'], description['pa100k']
    # elif experiment == 'rap':
    #     train_dataset = MultiLabelDataset(root='data_path',
    #                 label='train_list_path', transform=transform_train)
    #     val_dataset = MultiLabelDataset(root='data_path',
    #                 label='val_list_path', transform=transform_test)
    #     return train_dataset, val_dataset, attr_nums['rap'], description['rap']
    elif experiment == 'peta':
        train_dataset = MultiLabelDataset(split='train',
                    data_path='./dataset/peta/peta_partition.pkl', transform=transform_train)
        val_dataset = MultiLabelDataset(split='test',
                    data_path='./dataset/peta/peta_partition.pkl', transform=transform_train)
        return train_dataset, val_dataset, attr_nums['peta'], description['peta']
