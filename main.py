from __future__ import print_function
import copy
import  csv
import os
import numpy
import torch
import random
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import foolbox as fb 
from foolbox import PyTorchModel, accuracy, samples
from torch.autograd import Variable
import torch.nn.init as nninit
import torchvision.transforms.functional as TF
from vit import VisionTransformer
from torch.utils.data import Dataset
from PIL import Image
import eagerpy as ep
import pandas as pd

### Define what device we are using
use_cuda = True
print("CUDA Available: ",torch.cuda.is_available())
device = torch.device("cuda:1" if (use_cuda and torch.cuda.is_available()) else "cpu")
ROOT = '.'

            ### Define Required functions #### 

## Vision transformer module 
class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.model = VisionTransformer(
        img_size=32, patch_size=2, in_chans=3, num_classes=10, embed_dim=80, depth=20,
                 num_heads=20, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm)
    def forward(self,x):
        return self.model(x)

### Function for testing the model
def test(model,test_loader):
    model.eval()
    correct = 0
    avg_act = 0
    for data,target in test_loader:
        data = data.to(device)
        target = target.to(device)
        with torch.no_grad():
            out = torch.nn.Softmax(dim=1).cuda()(model(data))
                    
        act,pred = out.max(1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().cpu()
        avg_act += act.sum().data

    return 100. * float(correct) / len(test_loader),100. * float(avg_act) / len(test_loader.dataset)

## Our designed L-inf PGD attack( We have not used it for this experiment. Instead we went with the foolbox module.)
class pgd_attack():
    def __init__(self):
        pass

    def __call__(self, model, images, labels, eps, alpha, iters):
        adv_images = images.clone().detach().requires_grad_(True).to(images.device)
        labels = labels.to(device)
        loss = nn.CrossEntropyLoss()
        for i in range(iters):
            _adv_images = adv_images.clone().detach().requires_grad_(True)
            outputs = model(_adv_images)
            model.zero_grad()
            cost = loss(outputs, labels).to(device)
            cost.backward()
            grad=_adv_images.grad
            grad = grad.sign()

            assert (images.shape == grad.shape)

            adv_images = adv_images + grad * alpha
            delta = adv_images - images


            eps = eps / 255
            adv_images = images + ep.clip(adv_images - images, -eps, eps)
            adv_images = ep.clip(adv_images, 0,1).detach_()
            delta = adv_images - images
        return adv_images


        ### Test the models and select 100 correctly classified samples ###

## Load the data.
test_loader = torch.utils.data.DataLoader(torchvision.datasets.CIFAR10(root=ROOT, train=False, transform=transforms.Compose([
                        transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]), download=True),
                        batch_size=1,
                        shuffle=False,
                        num_workers=4
                        )


## Pretrained Model
pretrained_model = "./model/mdl.pth"
use_cuda = True

## Get the Architecture
model = NN()
model = model.to(device)

# Load the pretrained model
model.load_state_dict(torch.load(pretrained_model)["model"])

## Make the executable model for foolbox
mean = 0.5
std =0.5
lower_bound = (0-mean)/std
uppeer_bound = (1-mean)/std
fmodel = PyTorchModel(model.eval(), bounds=(lower_bound, uppeer_bound), device = device)

## Select 100 correctly calssified images
Success_images = np.zeros((100,3,32,32), dtype= np.float32)
Labels         = np.zeros((100), dtype= np.long)

index = 0
for data, target in test_loader:
    images = data.to(device)
    labels = target.to(device)

    with torch.no_grad():
        out = torch.nn.Softmax(dim=1).cuda()(model(images))
                
    act,pred = out.max(1, keepdim=True)
    correct  = pred.eq(labels.view_as(pred)).sum().cpu()

    if correct ==1: 
        Success_images[index] = images.cpu()
        Labels[index] = labels.cpu()
        index +=1
        if index == 100:
            break 

np.save('Correct_images.npy',Success_images)
np.save('Correct_labels.npy',Labels)
print("100 Correctly classified images are selected")
print("........")

### Dataloader for CIFAR10
class Cifar(Dataset):
    def __init__(self, image_file, label_file, shuffle=False):
        super(Cifar, self).__init__()
        self.image_numpy = np.load(image_file)
        self.labels = np.load(label_file)

    def __getitem__(self, index):

        ### Don't Use Data Transformation
        image = torch.from_numpy(np.array(self.image_numpy[index]))        
        label = torch.from_numpy(np.array(self.labels[index]))
        return image, label


    def __len__(self):
        return len(self.image_numpy)


## Test Loader 
test_clean_set = Cifar('Correct_images.npy', 'Correct_labels.npy')
testloader_clean = torch.utils.data.DataLoader(test_clean_set, batch_size=1, shuffle=False, num_workers=4)

## Test the model for saved samples
acc, _ = test(model, testloader_clean)
print("Test accuracy on those 100 samples: ", acc)

                    ## Now attack the model ###

## Attack Settings (we attack the model sequentially with different attacks)
epsilon = 5/255

attacks= [
fb.attacks.FGSM(),
fb.attacks.FGSM(),
fb.attacks.FGSM(),
fb.attacks.FGSM(),
fb.attacks.FGSM(),
fb.attacks.LinfBasicIterativeAttack(),
fb.attacks.LinfPGD(steps=5, abs_stepsize=1/255, random_start=True),
fb.attacks.LinfPGD(steps=5, abs_stepsize=epsilon, random_start=True),
fb.attacks.LinfPGD(steps=10, abs_stepsize=epsilon, random_start=True),
fb.attacks.LinfPGD(steps=15, abs_stepsize=epsilon, random_start=True),
fb.attacks.LinfPGD(steps=20, abs_stepsize=epsilon, random_start=True),
fb.attacks.LinfPGD(steps=30, abs_stepsize=epsilon, random_start=True),
]

## Saving directories
out_path_defended = "./Images/Defended_model/"
out_path_normal = "./Images/Normal_model/"
out_path_clean = "./Images/Clean_Images/"

## Inverse nrmalization
class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):

        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)

        return tensor

## For Denormalizaion
UnNorm = DeNormalize(mean=(0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))


## Attack Success Rate Function
def val_accuracy_calculation(model, data_load, device, model_type):
    model.eval()
    print('.....................')
    print('.....................')
    ## For defended model
    if model_type == 'Defended':

        ## Perturbation Norm
        epsilon = 10/255 

        image_numpy = np.load('Correct_images.npy')
        labels_original = np.load('Correct_labels.npy')
        correct = 0
        remove_keys = [k for k in range(100)]
        Success_images = np.zeros((100,3,32,32), dtype= np.float32)
        Labels = 11*np.ones((100), dtype= np.long)
        correct_label =11* np.ones((100), dtype= np.long)
        for_csv = {'File': [], 'Label': [], 'Prediction':[]}
        for i, attack in enumerate(attacks):
            for j, index in enumerate(remove_keys):

                ## Get the clean Image
                images_clean = torch.from_numpy(np.array(image_numpy[index]))
                images_clean = images_clean.view(1,3,32,32).to(device)
                labels = torch.from_numpy(np.array(labels_original[index]))
                labels = torch.tensor([labels])
                labels = labels.to(device)

                ## Perform the attack
                t1, adv_images, succ = attack(fmodel, images_clean, labels, epsilons=epsilon)

                ## Resize the perturbed images
                images_16 = torch.nn.functional.interpolate(adv_images, size=(16, 16), mode='bilinear', align_corners=False)

                ## Get the output of the model
                with torch.no_grad():
                    out = torch.nn.Softmax(dim=1).cuda()(model(adv_images)) 
                    out16x16 = torch.nn.Softmax(dim=1).cuda()(model(images_16))

                ## Get the prediction    
                act,pred = out.max(1, keepdim=True)
                _,pred16x16 = out16x16.max(1, keepdim=True)

                success = (pred16x16!=labels.view_as(pred16x16))[pred16x16==pred].sum().cpu()
                
                ## If image is adverarial remove the index 
                if success:
                    correct+=1

                    ## Check the perturbation size (if within the limit) 
                    perturbation_sizes = 255*abs((adv_images - images_clean)).max().cpu().numpy()
                    print("The calculated perturbation size:%2f"%(perturbation_sizes))
                    
                    ## Remove this clean image
                    remove_keys.remove(index)
                    

                    ## Save images 
                    image_name = out_path_defended+ "%d.png"%(index+1)
                    adv_images = UnNorm(adv_images)
                    torchvision.utils.save_image(adv_images,image_name,normalize=None)

                    for_csv['File'].append(image_name)
                    for_csv['Label'].append(int(labels.cpu().numpy()[0]))
                    for_csv['Prediction'].append(int(pred.cpu().numpy()[0]))



        data_frame = pd.DataFrame(
            data={'File': for_csv['File'], 'Label': for_csv['Label'], 'Prediction': for_csv['Prediction']},
            index=range(correct))
        data_frame.to_csv("Adversarial_Images_Defended_model.csv", index_label='Index')

        Accuracy = correct / 100
        print('....................')
        print('For %s Model, the Attack Success Rate is: %4f'%(model_type, Accuracy * 100))
        print('....................')
        print('....................')

    ## For Normallly Trained model
    else:
        ## perturbation norm 
        epsilon = 4/255

        image_numpy = np.load('Correct_images.npy')
        labels_original = np.load('Correct_labels.npy')

        for_csv = {'File': [], 'Label': [], 'Prediction':[]}
        
        Success_images_normal = np.zeros((100,3,32,32), dtype= np.float32)

        Labels = 11*np.ones((100), dtype= np.long)
        correct_label = 11*np.ones((100), dtype= np.long)
        correct = 0
        remove_keys = [k for k in range(100)]
        for i, attack in enumerate(attacks):
            for j, index in enumerate(remove_keys):

                ## Get the clean Image
                images_clean = torch.from_numpy(np.array(image_numpy[index]))
                images_clean = images_clean.view(1,3,32,32).to(device)
                labels = torch.from_numpy(np.array(labels_original[index]))
                labels = torch.tensor([labels])
                labels = labels.to(device)

                ## Perform the attack
                t1, adv_images , success = attack(fmodel, images_clean, labels, epsilons=epsilon)
                
                ## Get the output of the model
                with torch.no_grad():
                    out = torch.nn.Softmax(dim=1).cuda()(model(adv_images)) 
                
                ## Get the prediction           
                act,pred = out.max(1, keepdim=True)
                success = (pred!=labels.view_as(pred)).sum().cpu()
                
                ## If image is adverarial, remove the index 
                if success:
                    correct+=1

                    ## Check perturbation Size
                    perturbation_sizes = 255*abs((adv_images - images_clean)).max().cpu().numpy()
                    print("The calculated perturbation size:%2f"%(perturbation_sizes))
                    
                    ## Remove this clean image
                    remove_keys.remove(index)

                    ## Save images
                    adv_images = UnNorm(adv_images)
                    image_name = out_path_normal+ "%d.png"%(index+1)
                    torchvision.utils.save_image(adv_images,image_name,normalize=None)
                    
                    for_csv['File'].append(image_name)
                    for_csv['Label'].append(int(labels.cpu().numpy()[0]))
                    for_csv['Prediction'].append(int(pred.cpu().numpy()[0]))

                ## save clean images
                clean_images = UnNorm(images_clean)
                image_name = out_path_clean+ "%d.png"%(index+1)
                torchvision.utils.save_image(clean_images,image_name,normalize=None)

                
        data_frame = pd.DataFrame(
            data={'File': for_csv['File'], 'Label': for_csv['Label'], 'Prediction': for_csv['Prediction']},
            index=range(correct))
        data_frame.to_csv("Adversarial_Images_Normal_model.csv", index_label='Index')

        Accuracy = correct / 100
        print('....................')
        print('For %s Trained Model, the Attack Success Rate is: %4f'%(model_type, Accuracy * 100))
        print('....................')
        print('....................')


## Validate our attack
model_type = ['Normally', 'Defended']
for type_1 in model_type:
    val_accuracy_calculation(model, testloader_clean, device, type_1)

print('...................')
print('...................')


    ### Evaluate attack success rate for saved adversarial images ##
print("Evaluation on the saved adversarial images (The success rate should be 100)")
print(".............................")



### Dataloader for CIFAR10
class Cifar_csv(Dataset):
    def __init__(self, csv_file,transform, shuffle=False):
        super(Cifar_csv, self).__init__()

        self.data_df = pd.read_csv(csv_file)
        self.data = self.data_df['File']
        self.true_label = 'Label'
        self.predicted_label = 'Prediction'
        self.transform= transform

    def __getitem__(self, index):
        img = Image.open(self.data[index])
        img = self.transform(img)

        label = torch.from_numpy(np.array(self.data_df.iloc[index][self.true_label]))  
        pred = torch.from_numpy(np.array(self.data_df.iloc[index][self.predicted_label]))    

        return img, label

    def __len__(self):
        return len(self.data_df)

## Function for testing defended model
def test_defended(model,test_loader):
    model.eval()
    correct = 0
    avg_act = 0
    for data,target in test_loader:
        data = data.to('cuda:1')
        target = target.to('cuda:1')
        data16x16 = torch.nn.functional.interpolate(data, size=(16, 16),mode='bilinear', align_corners=False)
        
        with torch.no_grad():
            out = torch.nn.Softmax(dim=1).cuda()(model(data)) 
            out16x16 = torch.nn.Softmax(dim=1).cuda()(model(data16x16))
                    
        act,pred = out.max(1, keepdim=True)
        _,pred16x16 = out16x16.max(1, keepdim=True)
        correct += (pred16x16!=target.view_as(pred16x16))[pred16x16==pred].sum().cpu()  ## condition for success 
        avg_act += act.sum().data

    return 100. * float(correct) /len(test_loader)

## Tranformation
transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


## Validate saved images for Normal Model
csv_filename = "Adversarial_Images_Normal_model.csv"
test_clean_set = Cifar_csv(csv_filename, transform_test)
testloader_clean = torch.utils.data.DataLoader(test_clean_set, batch_size=1, shuffle=False, num_workers=4)

### Get the accuracy
acc,_ = test(model, testloader_clean)
print('Check Normally trained model')
print("The attack success rate for the saved adversarial examples:", 100-acc)
print('...........')
print('...........')

## Validate saved images for Defended Model 
csv_filename = "Adversarial_Images_Defended_model.csv"
test_clean_set = Cifar_csv(csv_filename, transform_test)
testloader_clean = torch.utils.data.DataLoader(test_clean_set, batch_size=1, shuffle=False, num_workers=4)
success_rate= test_defended(model, testloader_clean)
print("Now, lets check defended model")
print("The attack success rate for the saved adversarial examples:", success_rate)