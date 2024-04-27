import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader
import os
import pandas as pd
from tqdm import tqdm
import pickle

class CelebADataset(Dataset):
    def __init__(self, data, labels, classes, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform
        self.classes = list(classes)
        self.n_classes = len(self.classes)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = Image.open(self.data[idx]).convert('RGB')
        label = torch.Tensor(self.labels[idx])
        if self.transform:
            img = self.transform(img)
        sample = {'image': img, 'label': label}
        return sample
    

def get_loader(train_dataset, val_dataset, test_dataset, batch_size, device):
    num_workers = 0 if device.type == 'cuda' else 2
    pin_memory = True if device.type == 'cuda' else False

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory,shuffle=True)
    val_loader = DataLoader(val_dataset, num_workers=num_workers, pin_memory=pin_memory, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, num_workers=num_workers, pin_memory=pin_memory, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def create_celeba_dataset(dataset_path):
    f = open(os.path.join(dataset_path, 'list_attr_celeba.txt'))
    file = f.readlines()
    sample = int(file[0])
    classes = file[1].split(' ')
    classes.pop(-1)
    attr = []
    for i in file[2:]:
        list_ = i.split()
        list_.pop(0)
        list_ = list(map(int, list_))
        attr.append(list_)

    for li in attr:
        for ind in range(len(li)):
            if(li[ind] == -1):
                li[ind] = 0

    ########
    f = open(os.path.join(dataset_path, 'list_eval_partition.txt'))
    file = f.readlines()
    eval_dict = {'name': [],
            'eval': []}
    for i in file:
        key, value = i.split()
        eval_dict['name'].append(key)
        eval_dict['eval'].append(int(value))
    eval_dict_csv = pd.DataFrame(eval_dict)

    #######
    df = pd.DataFrame(attr, columns=classes)
    df.to_csv(os.path.join(dataset_path, 'attribute.csv'), index=False)
    eval_dict_csv.to_csv(os.path.join(dataset_path, 'eval.csv'), index=False)

    image_folder_path = os.path.join(dataset_path, "img_align_celeba")
    label_path = os.path.join(dataset_path, "attribute.csv")
    eval_path = os.path.join(dataset_path, "eval.csv")

    eval_list = pd.read_csv(eval_path)['eval'].values  
    eval_name = pd.read_csv(eval_path)['name'].values
    labels = pd.read_csv(label_path).values

    #
    indx, indy, recall = [0]*3, [0]*3, 0
    for i in eval_list:
        if recall == i - 1:
            recall = i
            indy[recall] += indy[recall - 1] + 1
            indx[recall] = indy[recall]
        else:
            indy[recall] += 1
    #
    train_list = [os.path.join(image_folder_path, name) for name in eval_name[indx[0]:]]
    train_label_list = labels[indx[0]:]
    val_list = [os.path.join(image_folder_path, name) for name in eval_name[indx[1]:]]
    val_label_list = labels[indx[1]:]
    test_list = [os.path.join(image_folder_path, name) for name in eval_name[indx[2]:]]
    test_label_list = labels[indx[2]:]

    data_transform=transforms.Compose([
        transforms.CenterCrop((178, 178)),
        # transforms.Resize((64, 64)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.5, 0.5, 0.5],
        #                     std=[0.5, 0.5, 0.5])
    ])

    classes = pd.read_csv(label_path).columns

    train_dataset = CelebADataset(train_list, train_label_list, classes, data_transform)
    val_dataset = CelebADataset(val_list, val_label_list, classes, data_transform)
    test_dataset = CelebADataset(test_list, test_label_list, classes, data_transform)

    return train_dataset, val_dataset, test_dataset


def get_dataset(dataset_name, dataset_path):
    if dataset_name == "celeba":
        return create_celeba_dataset(dataset_path)
    else:
        pass


def get_forget_and_retain_dataloaders(train_dataset, val_dataset, test_dataset, args, classes_to_forget):    
    retain_samples = []
    forget_samples = []
    retain_samples_val = []
    forget_samples_val = []
    retain_samples_test = []
    forget_samples_test = []

    from_cache = (False and len(classes_to_forget) == 1)
    if from_cache == False:
        for i, data in enumerate(tqdm(train_dataset)):
            is_forget = False
            for cls in classes_to_forget:
                if data['label'][cls] == 1:
                    is_forget = True
                    forget_samples.append(data)
                    break
            if is_forget == False:
                retain_samples.append(data)

        for data in tqdm(val_dataset):
            is_forget = False
            for cls in classes_to_forget:
                if data['label'][cls] == 1:
                    is_forget = True
                    forget_samples_val.append(data)
                    break
            if is_forget == False:
                retain_samples_val.append(data)

        for data in tqdm(test_dataset):
            is_forget = False
            for cls in classes_to_forget:
                if data['label'][cls] == 1:
                    is_forget = True
                    forget_samples_test.append(data)
                    break
            if is_forget == False:
                retain_samples_test.append(data)
                
    else:
        with open('lists_data.pkl', 'rb') as file:
            loaded_lists = pickle.load(file)

        retain_samples, forget_samples, forget_samples_val, retain_samples_val, forget_samples_test, retain_samples_test = loaded_lists

    print(f"retain: {len(retain_samples)}, forget: {len(forget_samples)}\n retain val: {len(retain_samples_val)}, forget val: {len(forget_samples_val)}\n retain test: {len(retain_samples_test)}, forget test: {len(forget_samples_test)}\n")    

    batch_size = args.batch_size
    
    retain_dl = DataLoader(retain_samples, batch_size, num_workers=0, pin_memory=True)
    forget_dl = DataLoader(forget_samples, batch_size, num_workers=0, pin_memory=True)
    forget_val_dl = DataLoader(forget_samples_val, batch_size, num_workers=0, pin_memory=True)
    retain_val_dl = DataLoader(retain_samples_val, batch_size, num_workers=0, pin_memory=True)
    forget_test_dl = DataLoader(forget_samples_test, batch_size, num_workers=0, pin_memory=True)
    retain_test_dl = DataLoader(retain_samples_test, batch_size, num_workers=0, pin_memory=True)
    
    return  retain_dl, forget_dl, retain_val_dl, forget_val_dl, retain_test_dl, forget_test_dl, retain_samples