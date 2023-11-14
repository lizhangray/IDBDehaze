import shutil
import os

import torch
import torchvision
from datasets import  tqdm

from torch import clamp
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from piqa import SSIM, PSNR


import networks
import util



class DualDehaze:
    def __init__(self,device,upsample_factor):
        self.device=device
        
        self.branchA=networks.knowledge_adaptation_UNet().to(self.device)
        self.branchB=networks.knowledge_adaptation_UNet().to(self.device)
        self.fuseAandB=networks.FuseNET(channel_in=28*2).to(self.device)
        self.SR_U=networks.RCAN(input_channel=28*2,scale=int(upsample_factor),num_features=64,num_rg=1,num_rcab=5,reduction=16).to(self.device)
        self.criterion_SSIM = SSIM()
        self.criterion_PSNR = PSNR()


        
            
    def loadmodel(self,model_name):
        path="model/"+model_name+".pth"
        checkpoint=torch.load(path)
        self.branchA.load_state_dict(checkpoint["branchA"])
        self.branchB.load_state_dict(checkpoint["branchB"])
        self.SR_U.load_state_dict(checkpoint["SR_U"])
        self.branchA.to(self.device)
        self.branchB.to(self.device)
        self.SR_U.to(self.device)
        


    def save(self,name):
        
        modelpath = "model/"+name+".pth"
        state = {'branchA': self.branchA.state_dict(), 'branchB': self.branchB.state_dict(),
                     'SR_U': self.SR_U.state_dict()}
        torch.save(state, modelpath)
         

    def eval(self, dataset):
        eval_dataloader = DataLoader(dataset=dataset, batch_size=1)
        psnr_list = []
        ssim_list = []
        device = torch.device("cuda")
        path_Dehaze = "result/"
        for index, data in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader),
                                desc='evaling Process'):
            torch.cuda.empty_cache()   
            inputs, labels ,labels_name= data
            inputs = inputs.to(device)
            labels = labels.to(device)
            labels_name=labels_name[0].split('.')[0]+".png"
            with torch.no_grad():
                self.branchA.eval()
                self.branchB.eval()
                self.SR_U.eval()
                centerCorp = torchvision.transforms.CenterCrop((inputs.shape[-2], inputs.shape[-1]))
                inputs = util.pad2affine(inputs, mod=32)
                A, feaA1 = self.branchA.to(device)(inputs)
                B, feaB1 = self.branchB.to(device)(inputs)
                D,_ = self.SR_U.to(device)(torch.cat((feaA1, feaB1), dim=1))
                centerCorp_HR = torchvision.transforms.CenterCrop((labels.shape[-2], labels.shape[-1]))
                D = centerCorp_HR(D)
                D=clamp(D,min=0,max=1)
                labels=clamp(labels,min=0,max=1)
                psnr_list.extend([self.criterion_PSNR(D, labels)])
                ssim_list.extend([self.criterion_SSIM(D.to("cpu"), labels.to("cpu"))])
                save_image(D, path_Dehaze + labels_name)
        avr_psnr = sum(psnr_list) / len(psnr_list)
        avr_ssim = sum(ssim_list) / len(ssim_list)
        print("psnr:{0},ssim{1}".format(avr_psnr, avr_ssim))   