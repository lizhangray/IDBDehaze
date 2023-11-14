
import torch
from torch.utils.data import ConcatDataset
import argparse

import data
from DualDehaze import DualDehaze

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='IDBNet')
    parser.add_argument('--dataset_hazy', help='Set the path to hazy input', type=str,required=True)
    parser.add_argument('--dataset_clean', help='Set the path to clean images', type=str,required=True)
    parser.add_argument('--model_name', help="Set the name to model file, model file shoud be placed in 'model' dir", type=str,required=True)
    parser.add_argument('--downsample_factor', help="Set upsacle factor form '1','0.5','0.25','1' means no any downsample to input and diractly output orianl size of the input image", type=float,required=True)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # you should have an gpu or you should modify some code.


    test_dataset_dehaze = data.EvalData_haze(args.dataset_hazy,args.dataset_clean,downsample_factor=args.downsample_factor)
    test_dataset=ConcatDataset([test_dataset_dehaze])
    
    upsample_factor=1
    if args.downsample_factor==0.5: 
       upsample_factor=2
    elif args.downsample_factor==0.25:
       upsample_factor=4
    
    model=DualDehaze(device=device,upsample_factor=upsample_factor)
    model.loadmodel(model_name=args.model_name)
    model.eval(test_dataset)