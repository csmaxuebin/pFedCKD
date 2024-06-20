from torchvision import datasets, transforms
import torch
from model import  ResNet, MobileNet, customResNet, customMobileNet162, customMobileNet138, customMobileNet150

def Load_Model(args):
    
    '''

    Function to load the required architecture (model) for federated learning

    '''


    if args.model == 'MobileNet':
        net_glob = MobileNet(args=args).to(args.device)
    elif args.model == 'ResNet':
        net_glob = ResNet.ResNet34(args=args).to(args.device)
    elif args.model =='customMobileNet162':
        net_glob = customMobileNet162(args=args).to(args.device)
    elif args.model =='customMobileNet150':
        net_glob = customMobileNet150(args=args).to(args.device)
    elif args.model =='customMobileNet138':
        net_glob = customMobileNet138(args=args).to(args.device)
    elif args.model =='customResNet204':
        net_glob = customResNet.customResNet204(args=args).to(args.device)
    elif args.model =='customResNet192':
        net_glob = customResNet.customResNet192(args=args).to(args.device)
    else:
        exit('Error: unrecognized model')
    
    return net_glob