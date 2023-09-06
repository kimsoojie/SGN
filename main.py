# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import argparse
import time
import shutil
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import os.path as osp
import csv
import numpy as np

np.random.seed(1337)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
from model import SGN
from model import CLIP
from data import NTUDataLoaders, AverageMeter
import fit
from util import make_dir, get_num_classes
from util import get_ntu120_action_classes
import yaml
import torch.nn.functional as F
import random

with open('config.yaml', 'r') as file:
    cfgs = yaml.safe_load(file)

parser = argparse.ArgumentParser(description='Skeleton-Based Action Recgnition')
fit.add_fit_args(parser)
parser.set_defaults(
    network=cfgs['cfgs']['network'],
    dataset = cfgs['cfgs']['dataset'],
    case = cfgs['cfgs']['case'],
    batch_size=cfgs['cfgs']['batch_size'],
    max_epochs=cfgs['cfgs']['max_epochs'],
    monitor=cfgs['cfgs']['monitor'],
    lr=cfgs['cfgs']['lr'],
    weight_decay=cfgs['cfgs']['weight_decay'],
    lr_factor=cfgs['cfgs']['lr_factor'],
    workers=cfgs['cfgs']['workers'],
    print_freq = cfgs['cfgs']['print_freq'],
    train = cfgs['cfgs']['train'],
    seg = cfgs['cfgs']['seg'], # skeleton sequence segmentation, randomly select one frame
    )
args = parser.parse_args()

def main():

    args.num_classes = get_num_classes(args.dataset)
    model = SGN(args.num_classes, args.dataset, args.seg, args)
    #model.load_state_dict(torch.load('results_cs/NTU120/SGN/0_best.pth')['state_dict'])
    
    model_clip = CLIP()
    
    ntu120_action_classes = get_ntu120_action_classes()
    ntu120_text_tokens = model_clip.tokenize(ntu120_action_classes)
    #print(ntu120_text_tokens.shape) #[120,77]
    
    total = get_n_params(model)
    print(model)
    print('The number of parameters: ', total)
    print('The modes is:', args.network)

    if torch.cuda.is_available():
        print('It is using GPU!')
        model = model.cuda()
            
    criterion = LabelSmoothingLoss(args.num_classes, smoothing=0.1).cuda()
    criterion_clip=CLIPLoss(model_clip,ntu120_text_tokens).cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.monitor == 'val_acc':
        mode = 'max'
        monitor_op = np.greater
        best = -np.Inf
        str_op = 'improve'
    elif args.monitor == 'val_loss':
        mode = 'min'
        monitor_op = np.less
        best = np.Inf
        str_op = 'reduce'

    scheduler = MultiStepLR(optimizer, milestones=[60, 90, 110], gamma=0.1)
    # Data loading
    ntu_loaders = NTUDataLoaders(args.dataset, args.case, seg=args.seg)
    train_loader = ntu_loaders.get_train_loader(args.batch_size, args.workers)
    val_loader = ntu_loaders.get_val_loader(args.batch_size, args.workers)
    train_size = ntu_loaders.get_train_size()
    val_size = ntu_loaders.get_val_size()


    test_loader = ntu_loaders.get_test_loader(32, args.workers)

    print('Train on %d samples, validate on %d samples' % (train_size, val_size))

    best_epoch = 0
    output_dir = make_dir(args.dataset)

    save_path = os.path.join(output_dir, args.network)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    checkpoint = osp.join(save_path, '%s_best.pth' % args.case)
    earlystop_cnt = 0
    csv_file = osp.join(save_path, '%s_log.csv' % args.case)
    log_res = list()

    lable_path = osp.join(save_path, '%s_lable.txt'% args.case)
    pred_path = osp.join(save_path, '%s_pred.txt' % args.case)

    # Training
    if args.train ==1:
        for epoch in range(args.start_epoch, args.max_epochs):

            print(epoch, optimizer.param_groups[0]['lr'])

            t_start = time.time()
            train_loss, train_acc = train(train_loader, model, criterion, criterion_clip, optimizer, epoch, model_clip, ntu120_text_tokens)
            val_loss, val_acc = validate(val_loader, model, criterion, criterion_clip, model_clip, ntu120_text_tokens)
            log_res += [[train_loss, train_acc.cpu().numpy(),\
                         val_loss, val_acc.cpu().numpy()]]

            print('Epoch-{:<3d} {:.1f}s\t'
                  'Train: loss {:.4f}\taccu {:.4f}\tValid: loss {:.4f}\taccu {:.4f}'
                  .format(epoch + 1, time.time() - t_start, train_loss, train_acc, val_loss, val_acc))

            current = val_loss if mode == 'min' else val_acc

            ####### store tensor in cpu
            current = current.cpu()

            if monitor_op(current, best):
                print('Epoch %d: %s %sd from %.4f to %.4f, '
                      'saving model to %s'
                      % (epoch + 1, args.monitor, str_op, best, current, checkpoint))
                best = current
                best_epoch = epoch + 1
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best': best,
                    'monitor': args.monitor,
                    'optimizer': optimizer.state_dict(),
                }, checkpoint)
                earlystop_cnt = 0
            else:
                print('Epoch %d: %s did not %s' % (epoch + 1, args.monitor, str_op))
                earlystop_cnt += 1

            scheduler.step()

        print('Best %s: %.4f from epoch-%d' % (args.monitor, best, best_epoch))
        with open(csv_file, 'w') as fw:
            cw = csv.writer(fw)
            cw.writerow(['loss', 'acc', 'val_loss', 'val_acc'])
            cw.writerows(log_res)
        print('Save train and validation log into into %s' % csv_file)

    ### Test
    args.train = 0
    model = SGN(args.num_classes, args.dataset, args.seg, args)
    model = model.cuda()
    test(test_loader, model, checkpoint, lable_path, pred_path, model_clip, ntu120_text_tokens)

def train_clip(train_loader, model, criterion, criterion_clip, optimizer, epoch, model_clip, ntu120_text_tokens):
    losses = AverageMeter()
    acces = AverageMeter()
    model.train()
    
    k=0
    inputs_list120=[]
    target_list120=[]
    
    #test=np.zeros(120)
    #for i in range(0,len(train_loader.dataset)):
    #    inputs, target= train_loader.dataset[i]
    #    test[target]+=1
    #print(test)   
    
    while k < 120:
        rnd_idx = random.randint(0, len(train_loader.dataset)-1)
        inputs, target= train_loader.dataset[rnd_idx]
        if target == k:
            inputs_list120.append(inputs)
            target_list120.append(target)
            k+=1
    print(inputs_list120.shape)#[bs,120,20,75]
    print(target_list120.shape)#[bs,120]
        
    for i in range(0,len(train_loader.dataset)):
        output, sekeleton_embeddings = model(inputs.cuda())


    
def train(train_loader, model, criterion, criterion_clip, optimizer, epoch, model_clip, ntu120_text_tokens):
    losses = AverageMeter()
    acces = AverageMeter()
    model.train()
    
    for i, (inputs, target) in enumerate(train_loader):
        #input:[bs,20,75]
        output, sekeleton_embeddings = model(inputs.cuda()) #sekeleton_embeddings:[bs, 512, 1, 20]
        #print(target.shape)#[bs]
        #print(inputs.shape)#[bs,20,75]
        
        #target:[bs]
        #target = target.cuda(async = True)
        if cfgs['cfgs']['network']=='SGN_CLIP':
            loss = criterion_clip(sekeleton_embeddings, target)
            target = target.cuda(async = True)
            acc = accuracy_clip(sekeleton_embeddings.data, target, model_clip, ntu120_text_tokens)
        
        # measure accuracy and record loss
        if cfgs['cfgs']['network']=='SGN':
            target = target.cuda(async = True)
            acc = accuracy(output.data, target)
            loss=criterion(output, target)
        
        losses.update(loss.item(), inputs.size(0))
        acces.update(acc[0], inputs.size(0))

        # backward
        optimizer.zero_grad()  # clear gradients out before each mini-batch
        loss.backward()
        optimizer.step()

        if (i + 1) % args.print_freq == 0:
            print('Epoch-{:<3d} {:3d} batches\t'
                  'loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'accu {acc.val:.3f} ({acc.avg:.3f})'.format(
                   epoch + 1, i + 1, loss=losses, acc=acces))

    return losses.avg, acces.avg


def validate(val_loader, model, criterion, criterion_clip, model_clip, ntu120_text_tokens):
    losses = AverageMeter()
    acces = AverageMeter()
    model.eval()

    for i, (inputs, target) in enumerate(val_loader):
        with torch.no_grad():
            output, sekeleton_embeddings = model(inputs.cuda())
        target = target.cuda(async=True)
        
        
        with torch.no_grad():
            if cfgs['cfgs']['network']=='SGN_CLIP':
                loss = criterion_clip(sekeleton_embeddings, target)
                acc = accuracy_clip(sekeleton_embeddings.data, target, model_clip, ntu120_text_tokens)
            if cfgs['cfgs']['network']=='SGN':
                loss=criterion(output,target)
                acc = accuracy(output.data, target)
            
        # measure accuracy and record loss
        losses.update(loss.item(), inputs.size(0))
        acces.update(acc[0], inputs.size(0))

    return losses.avg, acces.avg


def test(test_loader, model, checkpoint, lable_path, pred_path,model_clip, ntu120_text_tokens):
    acces = AverageMeter()
    # load learnt model that obtained best performance on validation set
    model.load_state_dict(torch.load(checkpoint)['state_dict'])
    model.eval()

    label_output = list()
    pred_output = list()

    t_start = time.time()
    for i, (inputs, target) in enumerate(test_loader):
        with torch.no_grad():
            output, skeleton_embedding = model(inputs.cuda())
            if cfgs['cfgs']['network']=='SGN_CLIP':
                output=skeleton_embedding

            output = output.view((-1, inputs.size(0)//target.size(0), output.size(1)))
            output = output.mean(1)

        label_output.append(target.cpu().numpy())
        pred_output.append(output.cpu().numpy())

        if cfgs['cfgs']['network']=='SGN':
            acc = accuracy(output.data, target.cuda(async=True))
        if cfgs['cfgs']['network']=='SGN_CLIP':
            acc=accuracy_clip(skeleton_embedding.data,target, model_clip, ntu120_text_tokens)
        acces.update(acc[0], inputs.size(0))


    label_output = np.concatenate(label_output, axis=0)
    np.savetxt(lable_path, label_output, fmt='%d')
    pred_output = np.concatenate(pred_output, axis=0)
    np.savetxt(pred_path, pred_output, fmt='%f')

    print('Test: accuracy {:.3f}, time: {:.2f}s'
          .format(acces.avg, time.time() - t_start))


def accuracy(output, target):
    batch_size = target.size(0)
    _, pred = output.topk(1, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    correct = correct.view(-1).float().sum(0, keepdim=True)

    return correct.mul_(100.0 / batch_size)

def accuracy_clip(skeleton_embedding, target, model_clip, ntu120_text_tokens):
    batch_size = target.size(0)
    clip_text_embeddings=model_clip.encode_text(ntu120_text_tokens).t() #[512,120]
    clip_text_embeddings=clip_text_embeddings.unsqueeze(0).expand(batch_size, 512, 120)
    #print(clip_text_embeddings.shape)#[bs,512,120]
    
    skeleton_embedding = skeleton_embedding.view(20,-1,512) #[20,bs,512]
    skeleton_embedding=torch.mean(skeleton_embedding,dim=0) #[bs,512]
    #skeleton_embedding = skeleton_embedding.unsqueeze(2) #[bs,512,1]
    
    if batch_size != cfgs['cfgs']['batch_size']:
        correct=[1]
        correct[0]=0
        acc=torch.tensor(correct)
        print('=====BATCH_SIZE:',batch_size)
        return acc
    
    #cnt=0
    #for i in range(0, batch_size):
    #    target_text_emb = clip_text_embeddings[i,:,target[i]-1]
    #    skeleton_emb = skeleton_embedding[i,:]
    #    cosine_similarity=F.cosine_similarity(target_text_emb, skeleton_emb, dim=-1)
    #    if cosine_similarity > 0.9:
    #        cnt = cnt + 1
    
    cnt=0
    t=0
    for i in range(0, batch_size):
        arr=[]
        skeleton_emb = skeleton_embedding[i,:]
        _t=0
        for k in range(0,120):
            target_text_emb = clip_text_embeddings[i,:,k]
            cosine_similarity=F.cosine_similarity(target_text_emb, skeleton_emb, dim=-1)
            arr.append(cosine_similarity)
            if cosine_similarity > 0.9:
                _t+=1
        if _t > 5:
            t+=1
        topk_val, topk_idx = torch.tensor(arr).topk(5, -1, True, True)
        if target[i].cpu() in topk_idx:
            cnt = cnt+1
    #print(t,'/',batch_size)    
    
    correct=[1]
    correct[0]=100*(cnt/batch_size)
    acc=torch.tensor(correct)
    #print(np.where(classes > 0))
    
    return acc


def save_checkpoint(state, filename='checkpoint.pth.tar', is_best=False):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

class CLIPLoss(nn.Module):
    def __init__(self, model_clip, ntu120_text_tokens):
        super(CLIPLoss, self).__init__()
        self.model_clip=model_clip
        self.ntu120_text_tokens=ntu120_text_tokens
        
    def forward(self, skeleton_embedding, target):
        text_embeddings=self.model_clip.encode_text(self.ntu120_text_tokens).cpu() #[120,512]
        bs=len(target)
        text_features = np.zeros((bs,512))
        for i in range(0,bs):
            idx=target.cpu().numpy()[i]
            text_features[i,:]=text_embeddings[idx,:]
        skeleton_embedding = skeleton_embedding.view(-1,512) #[20*bs,512]
        text_features = torch.tensor(text_features).to(skeleton_embedding.device)
        
        cosine_similarity = F.cosine_similarity(text_features.unsqueeze(0), skeleton_embedding.unsqueeze(1), dim=-1) #[20*bs,bs]
        clip_loss = 1-cosine_similarity.view(-1).mean()
        loss = clip_loss
        
        #skeleton_embedding = skeleton_embedding.mean(dim=0)
        #cross_entropy=F.binary_cross_entropy_with_logits(text_features[target.cpu().numpy()[i],:], skeleton_embedding)
        #loss = cross_entropy
        
        return loss
    
if __name__ == '__main__':
    main()
    
