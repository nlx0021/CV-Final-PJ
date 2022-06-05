import argparse
from sched import scheduler
import numpy as np
from torch import is_vulkan_available, nn
import torch
import random
from tqdm import tqdm as tqdm
from os.path import join, exists
import matplotlib.pyplot as plt
import os
from torchtoolbox.tools import mixup_data, mixup_criterion
from torch.utils.tensorboard import SummaryWriter

from vit_pytorch import ViT


'''
from vit_pytorch.vit_for_small_dataset import ViT
'''

from model import resnet18
from data_loader import get_cifar_loader
from utils import rand_bbox
from utils import get_number_of_parameters

# Set a random seed to ensure reproducible results
def set_random_seeds(seed_value=0, device='cpu'):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    if device != 'cpu': 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train(model,
          optimizer,
          optimizer_finetune,
          lr,
          criterion,
          train_loader,
          val_loader,
          test_loader,
          device='cpu',
          scheduler=None,
          epochs_n=100,
          best_model_path=None,
          name='default',
          mixup=.2,
          LSUV=True,
          finetune=True,
          cutmix=None):
    
    # Check param.
    assert not (mixup and cutmix), "Do not use both mixup and cutmix."
    
    # Tensorboard.
    writer = SummaryWriter(join("tensorboard", name))
    
    # Mkdir.
    root = os.getcwd()
    dir_path = join(root, 'experiment', name)
    if exists(dir_path):
        print("The experiment dir exists, hence the data will be overwrited.")
    else:
        os.mkdir(dir_path)
    
    # Device.
    model.to(device)
    print('Device is %s' % device)
    
    # Curves.
    learning_curve = [np.nan] * epochs_n
    val_loss_curve = [np.nan] * epochs_n
    train_accuracy_curve = [np.nan] * epochs_n
    val_accuracy_curve = [np.nan] * epochs_n
    max_val_accuracy = 0
    max_val_accuracy_epoch = 0
    
    batches_n = len(train_loader)    
    losses_list = []
    
    # LSUV init.
    if LSUV:
        from external.LSUV import LSUVinit
        x, _ = next(iter(train_loader))
        x = x.to(device)
        model = LSUVinit(model, x, cuda=True)
            
    # Train.
    for epoch in tqdm(range(epochs_n), unit='epoch'):
        
        print('\n')
        
        model.train()          # Training mode.
        
        # To record the variables.
        loss_list = []
        learning_curve[epoch] = 0
        train_accuracy_curve[epoch] = 0 
        correct = 0
        
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            
            if mixup:
                x, y_a, y_b, lam = mixup_data(x, y, mixup)
                pred = model(x)
                loss = mixup_criterion(criterion, pred, y_a, y_b, lam)
                
            elif cutmix:
                beta, p = cutmix
                r = np.random.rand(1)
                if r < p:
                    lam = np.random.beta(beta, beta)
                    if device == 'cpu':
                        rand_index = torch.randperm(x.size()[0])
                    else:
                        rand_index = torch.randperm(x.size()[0]).cuda()
                    y_a = y
                    y_b = y[rand_index]
                    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
                    x[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]
                    # adjust lambda to exactly match pixel ratio
                    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
                    # compute output
                    pred = model(x)
                    loss = criterion(pred, y_a) * lam + criterion(pred, y_b) * (1. - lam)
                    # loss = distiller(x, y_a) * lam + distiller(x, y_b) * (1. - lam)
                else:
                    # compute output
                    pred = model(x)
                    loss = criterion(pred, y)                
                    # loss = distiller(x, y)    
            
            else:
                pred = model(x)
                loss = criterion(pred, y)
            
            # Record the variables.
            loss_list.append(loss.item())        
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            
            loss.backward()
            
            # If finetune:
            if finetune:
                assert optimizer_finetune is not None
                if epoch > finetune:
                    optimizer_finetune.step()
                else:
                    optimizer.step()
            else:
                optimizer.step()
        
        # Scheduler step.
        if scheduler is not None:
            # Only step in not-finetuning stage.
            if not finetune or epoch <= finetune:
                scheduler.step()                    
           
        # Recourd the variables for one epoch. 
        losses_list.append(loss_list)
        learning_curve[epoch] = sum(loss_list) / batches_n
        train_accuracy_curve[epoch] = correct / len(train_loader.dataset)
        
        print('The train loss of epoch %d is : %f' % (epoch, learning_curve[epoch]))
        print('The train accuracy of epoch %d is : %f' % (epoch, train_accuracy_curve[epoch]))
        
        writer.add_scalar("train_loss", sum(loss_list) / batches_n, epoch)
        writer.add_scalar("train_acc", correct / len(train_loader.dataset), epoch)
        
        # Val.
        model.eval()
        
        correct = 0
        loss_list = []
        val_accuracy_curve[epoch] = 0
        val_loss_curve[epoch] = 0
        
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)
            
            pred = model(x)
            loss_list.append(criterion(pred, y).item())
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            
        # Record the variables.
        correct /= len(val_loader.dataset)
        val_accuracy_curve[epoch] = correct
        val_loss_curve[epoch] = sum(loss_list) / len(val_loader)
        
        writer.add_scalar("val_loss", sum(loss_list) / len(val_loader), epoch)
        writer.add_scalar("val_acc", correct, epoch)
        
        if correct > max_val_accuracy:
            max_val_accuracy = correct
            max_val_accuracy_epoch = epoch
        
        print('The val accuracy of epoch %d is : %f' % (epoch, correct))
        
    # Test.
    model.eval()
    test_correct = 0
    for x, y in test_loader:
        x = x.to(device)
        y = y.to(device)
        
        pred = model(x)
        test_correct += (pred.argmax(1) == y).type(torch.float).sum().item()    
    
    test_correct /= len(test_loader.dataset)
    
    print("The test accuracy is: %f" % test_correct)
    
    # Count parameters.
    parameters_n = 0
    for parameter in model.parameters():
        parameters_n += np.prod(parameter.shape).item()
    
    # Summary.
    with open(join(dir_path, 'summary.txt'), 'w') as f:
        # Model.
        f.write('Model:\n\n')
        f.write('Model: ' + str(model) + '\n')
        # Parameters.
        f.write('Parameters:\n\n')
        f.write('Learning rate: %f\n' % lr)
        f.write('Total epoches: %d\n' % epochs_n)
        f.write('Loss function: ' + str(criterion) + '\n')
        # f.write('Scheduler: ' + str(scheduler.state_dict()) + '\n')
        # Curves.
        f.write('Curves:\n\n')
        f.write('learning_curve: ' + str(learning_curve) + '\n')
        f.write('val_loss_curve: ' + str(val_loss_curve) + '\n')
        f.write('train_accuracy_curve' + str(train_accuracy_curve) + '\n')
        f.write('val_accuracy_curve' + str(val_accuracy_curve) + '\n')
        f.write('max_val_accuracy: %f\n' % max_val_accuracy)
        f.write('max_val_accuracy_epoch: %d\n' % max_val_accuracy_epoch)
        # Test result.
        f.write('Test result:\n\n')
        f.write('Test accuracy: %f\n' % test_correct)
        # Parameters' number.
        f.write('The number of para:\n\n')
        f.write('para_n: %d\n' % parameters_n)
        
    # Plot the curves.
    plt.plot(range(epochs_n), learning_curve, label='loss in trainset', linewidth=3, color='r',
             marker='o', markerfacecolor='blue', markersize=5)
    plt.plot(range(epochs_n), val_loss_curve, label='loss in validset', linewidth=3, color='b',
             marker='o', markerfacecolor='red', markersize=5)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('losses')
    plt.legend()
    plt.savefig(join(dir_path, 'loss.png'))
    plt.clf() 
    
    plt.plot(range(epochs_n), train_accuracy_curve, label='accuracy in trainset', linewidth=3, color='r',
             marker='o', markerfacecolor='blue', markersize=5)
    plt.plot(range(epochs_n), val_accuracy_curve, label='accuracy in validset', linewidth=3, color='b',
             marker='o', markerfacecolor='red', markersize=5)
    plt.xlabel('Epochs')
    plt.ylabel('Acc')
    plt.title('Accuracy')
    plt.legend()
    plt.savefig(join(dir_path, 'Acc.png')) 
    plt.clf() 
    
    # Save the model.
    torch.save(model.state_dict(), join('./model', name+'.pth'))
    
    return learning_curve, val_accuracy_curve
    
    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='hw04')
    parser.add_argument('--experiment', 
                    type=str,
                    default='sb')
    parser.add_argument('--lr',
                    type=float,
                    default=2e-2)
    parser.add_argument('--epochs_n',
                    type=int,
                    default=200)
    args = parser.parse_args()
    
    
    names = [args.experiment]
    
    for i in range(len(names)):
        name = names[i]
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        lr = args.lr
        epochs_n = args.epochs_n
        LSUV = True
        ################################### Bench 2: Cutout. ##################################
        cutout = False    
        ################################### Bench 2: Cutout. ##################################                                    
        
        set_random_seeds(21, device=device)
        
        train_loader, val_loader, test_loader = get_cifar_loader(batch_size=128, n_items=-1, data_aug=True, cutout=cutout)
        print(len(train_loader.dataset), len(test_loader.dataset), len(val_loader.dataset))
        
        
        ################################### Transformer ##################################

        model = ViT(
            image_size = 32,
            patch_size = 4,
            num_classes = 100,
            dim = 512,
            depth = 7,
            heads = 8,
            mlp_dim = 512,
            dropout = 0,
            emb_dropout = 0
        )
        
        print(model)
        
        print('\n\n')
        print('number of parameters:', get_number_of_parameters(model))
        #######################################################################################
        
        # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        # optim_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 60, 70, 80, 90, 100], gamma=.5)
        # optim_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=.2)
        optim_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 120, 150, 180], gamma=.3)
        # optim_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=.7)
        
        # optim_scheduler = None
        
        criterion = nn.CrossEntropyLoss()
    
        train(model=model,
              optimizer=optimizer,
              optimizer_finetune=None,
              lr=lr,
              criterion=criterion,
              train_loader=train_loader,
              val_loader=val_loader,
              test_loader=test_loader,
              device=device,
              scheduler=optim_scheduler,
              epochs_n=epochs_n,
              name=name,
    ################################### Bench 1: Mixup. ##################################
              mixup=None,
    ################################### Bench 1: Mixup. ##################################     
    ################################### Bench 3: Cutmix. ##################################   
              cutmix=(1.0, .5),
              #cutmix=None,
    ################################### Bench 3: Cutmix. ##################################          
              LSUV=True,
              finetune=None)
