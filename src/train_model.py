#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torch.distributed as dist
import torch.nn.functional as F
import torch.utils.data
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

import argparse
import os
import sys
import json
import logging
import pip
import sys

#def import_or_install(package):
#    try:
#        __import__(package)
#    except ImportError:
#        os.execute(f"pip install {package}")

#required_packages=['smdebug']

#for package in required_packages:
#    import_or_install(package)
    
import smdebug.pytorch as smd

# If ImageFile.LOAD_TRUNCATED_IMAGES != True
# then the following error occurs
#File "/opt/conda/lib/python3.6/site-packages/PIL/ImageFile.py", line 247, in load
#    "(%d bytes not processed)" % len(b)
#OSError: image file is truncated (150 bytes not processed)
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)

num_classes = 133
image_size = 224

data_mean_std = {
    "mean": {
        "train": [0.4849703311920166, 0.4542146325111389, 0.3907787501811981], 
        "valid": [0.47780725359916687, 0.4509781301021576, 0.38741418719291687], 
        "test": [0.4849252998828888, 0.45060989260673523, 0.3871392011642456]
    }, 
    "std": {
        "train": [0.23013180494308472, 0.22563056647777557, 0.2250407487154007], 
        "valid": [0.22859683632850647, 0.2253820300102234, 0.22569558024406433], 
        "test": [0.2284841686487198, 0.22324755787849426, 0.22187553346157074]
    }
}


def test(model, test_loader, criterion, **kwargs):
    '''
    Test model
    '''
    use_cuda = kwargs['use_cuda']
    device = kwargs['device']
    smd_hook = kwargs['smd_hook']

    logger.debug(
        "Processes {}/{} ({:.0f}%) of test data".format(
            len(test_loader.sampler),
            len(test_loader.dataset),
            100.0 * len(test_loader.sampler) / len(test_loader.dataset),
        )
    )

    model.eval()
    smd_hook.set_mode(smd.modes.EVAL)
    
    test_loss = 0
    test_correct = 0
    
    with torch.no_grad():

        for inputs, labels in test_loader:
 
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            test_loss += loss.item() * inputs.size(0)
            test_correct += torch.sum(preds == labels.data)

    test_loss = test_loss / len(test_loader.dataset)

    logger.info(
        "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, test_correct, len(test_loader.dataset), 100.0 * test_correct / len(test_loader.dataset)
        )
    )


def train(model, train_loader, valid_loader, criterion, optimizer, **kwargs):
    '''
    Train using the model provided
    '''
    is_distributed = len(args.hosts) > 1 and args.backend is not None
    logger.debug("Distributed training - {}".format(is_distributed))
    use_cuda = kwargs['use_cuda']
    device = kwargs['device']
    smd_hook = kwargs['smd_hook']
    
    # for more information on distributed training see
    # https://sagemaker.readthedocs.io/en/stable/frameworks/pytorch/using_pytorch.html
    if is_distributed:
        # Initialize the distributed environment.
        world_size = len(args.hosts)
        os.environ["WORLD_SIZE"] = str(world_size)
        host_rank = args.hosts.index(args.current_host)
        dist.init_process_group(backend=args.backend, rank=host_rank, world_size=world_size)
        logger.info(
            "Initialized distributed environment: '{}' backend on {} nodes. ".format(
                args.backend, dist.get_world_size()
            )
            + "Current host rank is {}. Number of gpus: {}".format(dist.get_rank(), args.num_gpus)
        )

    # set the seed for generating random numbers
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)

    logger.debug(
        "Processes {}/{} ({:.0f}%) of train data".format(
            len(train_loader.sampler),
            len(train_loader.dataset),
            100.0 * len(train_loader.sampler) / len(train_loader.dataset),
        )
    )

    input_size = image_size 

    if is_distributed and use_cuda:
        # multi-machine multi-gpu case
        model = torch.nn.parallel.DistributedDataParallel(model)
    else:
        # single-machine multi-gpu case or single-machine or multi-machine cpu case
        model = torch.nn.DataParallel(model)

    # used to test if model has improved
    # only save if it has
    valid_loss_min = np.Inf 

    for epoch in range(1, args.epochs + 1):
        model.train()
        smd_hook.set_mode(smd.modes.TRAIN)

        train_loss = 0.0
        valid_loss = 0.0
        train_corrects = 0
        valid_corrects = 0

        for batch_idx, (data, target) in enumerate(train_loader, 1):
            # move to GPU if available
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            # forward
            output = model(data)
            loss = criterion(output, target)
            
            _, preds = torch.max(output, 1)

            loss.backward()
            
            if is_distributed and not use_cuda:
                # average gradients manually for multi-machine cpu case only
                _average_gradients(model)
                
            optimizer.step()
            # statistics
            train_loss += loss.item() * data.size(0)
            train_corrects += torch.sum(preds == target.data)
            
            if batch_idx % args.log_interval == 0:
                logger.info(
                    "Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}".format(
                        epoch,
                        batch_idx * len(data),
                        len(train_loader.sampler),
                        100.0 * batch_idx / len(train_loader),
                        loss.item(),
                    )
                )

        train_loss = train_loss / len(train_loader.dataset)
        train_acc = train_corrects.double() / len(train_loader.dataset)
        
        logger.info('Train Loss: {:.4f} Acc: {:.4f}'.format(train_loss, train_acc))
        
        # validate loss
        model.eval()
        smd_hook.set_mode(smd.modes.EVAL)

        for batch_idx, (data, target) in enumerate(valid_loader, 1):

            # move to GPU if available
            data, target = data.to(device), target.to(device)

            # update the average validation loss
            
            # forward
            output = model(data)
            loss = criterion(output, target)
            _, preds = torch.max(output, 1)
            
            valid_loss += loss.item() * data.size(0)
            valid_corrects += torch.sum(preds == target.data)
            
            if batch_idx % args.log_interval == 0:
                logger.info(
                    "Validate Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}".format(
                        epoch,
                        batch_idx * len(data),
                        len(valid_loader.sampler),
                        100.0 * batch_idx / len(valid_loader),
                        loss.item(),
                    )
                )

        valid_loss = valid_loss / len(valid_loader.dataset)
  
        # save the model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print(f'Model improved - validation loss decreased from {valid_loss_min} to {valid_loss}.\nSaving Model')
            save_model(model, args.model_dir)
            valid_loss_min = valid_loss
    return model
    

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
    
    
def net(use_pretrained=True, feature_extract=False):
    """ Using Squeezenet
    """
    model_ft = models.squeezenet1_0(pretrained=use_pretrained)
    set_parameter_requires_grad(model_ft, feature_extract)
    model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
    model_ft.num_classes = num_classes
    return model_ft


def model_fn(model_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_extract = False
    use_pretrained = False
    model = net(feature_extract=feature_extract, use_pretrained=use_pretrained)
    model = torch.nn.DataParallel(model)
    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        model.load_state_dict(torch.load(f))
    return model.to(device)
    

def predict_fn(input_data, model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    with torch.no_grad():
        return model(input_data.to(device))

    
def get_mean_std_from_images():
    '''
    Returns pre-calculated means and standard deviations
    for each dataset - train, valid, test
    
    '''
    return data_mean_std
    #fname = 'dog_images_mean_std.json'
    #if path.exists(fname):
    #    with open(fname) as json_file:
    #        info = json.load(json_file)        
    #        return info


def create_data_loaders(data_dir, batch_size, test_batch_size):
    '''
    '''
    mean_std = get_mean_std_from_images()
    
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            transforms.Normalize(mean_std['mean']['train'], mean_std['std']['train'])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            transforms.Normalize(mean_std['mean']['valid'], mean_std['std']['valid'])

        ]),
        'test': transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean_std['mean']['test'], mean_std['std']['test'])

        ]) 
    }
    

    print("Initializing Datasets and Dataloaders...")

    # Create training, validation and test datasets
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'valid', 'test']}
    # Create training, validation and test dataloaders
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], 
                                                       batch_size=test_batch_size if x == 'test' else batch_size, 
                                                       shuffle=True, 
                                                       num_workers=4) for x in ['train', 'valid', 'test']}
    return dataloaders_dict

def save_model(model, model_dir):
    logger.info("Saving the model.")
    path = os.path.join(model_dir, "model.pth")
    # recommended way from http://pytorch.org/docs/master/notes/serialization.html
    torch.save(model.cpu().state_dict(), path)


def main(args):
    model=net()
    
    loss_criterion = nn.CrossEntropyLoss()
    
    # Create and register the hook
    smd_hook = smd.Hook.create_from_json_file()
    smd_hook.register_module(model)
    # note register_hook just calls register_module i.e. they are the same
    # smd_hook.register_hook(model)
    # as per https://github.com/awslabs/sagemaker-debugger/blob/master/docs/pytorch.md
    # If using a loss which is a subclass of nn.Module, call hook.register_loss(loss_criterion) once before starting training.
    # which in this case nn.CrossEntropyLoss is (CrossEntropyLoss -> _WeightedLoss -> _Loss -> Module)
    smd_hook.register_loss(loss_criterion)
    
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    elif args.optimizer == 'asgd':
        optimizer = optim.ASGD(model.parameters(), lr=args.lr,
                                     lambd=0.0001, 
                                     alpha=0.75, 
                                     t0=1000000.0, 
                                     weight_decay=0)
    elif args.optimizer == 'adam':
        optimizer = optim.Adamax(model.parameters(), lr=args.lr, 
                                 betas=(0.9, 0.999), 
                                 eps=1e-08, 
                                 weight_decay=0)
    elif args.optimizer == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr, 
                                  alpha=0.99, eps=1e-08, 
                                  weight_decay=0, 
                                  momentum=args.momentum, 
                                  centered=False)

    data_loaders = create_data_loaders(args.data_dir, args.batch_size, args.test_batch_size)    
    use_cuda = args.num_gpus > 0
    # Detect if we have a GPU available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    kwargs = {"num_workers": 1, 
              "device": device,
              "use_cuda": use_cuda,
              "smd_hook": smd_hook}

    model=train(model, data_loaders['train'], data_loaders['valid'], loss_criterion, optimizer, **kwargs)
    
    test(model, data_loaders['test'], loss_criterion, **kwargs)
    
    save_model(model, args.model_dir)

    
    
if __name__=='__main__':
    logger.setLevel(int(os.environ.get('SM_LOG_LEVEL', logging.INFO)))

    parser = argparse.ArgumentParser()
    
    # hyperparameters sent by the client are passed as command-line arguments to the script.
    
    parser.add_argument('--epochs', type=int, default=int(os.environ['SM_HP_EPOCHS']))
    hyperparameters = json.loads(os.environ['SM_HPS'])
    
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--test-batch-size', type=int, default=100)

    parser.add_argument('--use-cuda', type=bool, default=False)

    # Data, model, and output directories
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument("--data-dir", type=str, default=os.environ["SM_CHANNEL_TRAINING"])
    parser.add_argument("--optimizer", type=str, default='sgd')

    # Container environment
    parser.add_argument("--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"]))
    parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument("--num-gpus", type=int, default=os.environ["SM_NUM_GPUS"])

    hyperparameters = json.loads(os.environ['SM_HPS'])
    logger.info(f'hyperparameters {json.dumps(hyperparameters)}')

    args, _ = parser.parse_known_args()

    parser.add_argument(
        "--lr", type=float, default=0.01, metavar="LR", help="learning rate (default: 0.01)"
    )
    parser.add_argument(
        "--momentum", type=float, default=0.5, metavar="M", help="SGD momentum (default: 0.5)"
    )
    parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
    parser.add_argument(
        "--log-interval",
        type=int,
        default=100,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument("--backend", type=str, default=os.environ["SM_HP_BACKEND"])

    args=parser.parse_args()
    
    main(args)
