# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import argparse
import time
import shutil
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import os.path as osp
import csv
import numpy as np

#np.random.seed(1337)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
from model import SGN
from model import CLIP
from data import NTUDataLoaders, AverageMeter
import fit
from util import make_dir, get_num_classes
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
    model.load_state_dict(torch.load('results_cv/NTU120/SGN/1_best.pth')['state_dict'])
    
    model_clip = CLIP()
    
    total_clip = get_n_params(model_clip)
    print(model_clip)
    print('The number of parameters (CLIP): ', total_clip)
    
    total = get_n_params(model)
    print(model)
    print('The number of parameters: ', total)
    print('The modes is:', args.network)

    if torch.cuda.is_available():
        print('It is using GPU!')
        model = model.cuda()
        model_clip = model_clip.cuda()
        if torch.cuda.device_count() > 1:
            model_clip = nn.DataParallel(model_clip)
            
    criterion = LabelSmoothingLoss(args.num_classes, smoothing=0.1).cuda()
    
    if cfgs['cfgs']['clip_train'] == False:
        criterion_clip=CLIPLoss(model_clip,type='cross_entropy').cuda()
    elif cfgs['cfgs']['clip_train'] == True:
        criterion_clip=CLIPTrainLoss().cuda()
    
        
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer_clip=None
    if cfgs['cfgs']['clip_train'] == True:
        optimizer_clip = optim.Adam(model_clip.parameters(), lr=5e-7,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2)
    
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
    if cfgs['cfgs']['clip_train'] == True:
        scheduler_clip = MultiStepLR(optimizer_clip, milestones=[60, 90, 110], gamma=0.1)
    
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
    checkpoint_clip = osp.join(save_path, '%s_best_clip.pth' % args.case)
    
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
            train_loss, train_acc, train_loss_clip, train_acc_clip = train(train_loader, model, criterion, criterion_clip, optimizer, epoch, model_clip, optimizer_clip=optimizer_clip)
            val_loss, val_acc, val_loss_clip, val_acc_clip = validate(val_loader, model, criterion, criterion_clip, model_clip)
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
                if cfgs['cfgs']['clip_train'] == True:
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'state_dict': model_clip.state_dict(),
                        'best': best,
                        'monitor': args.monitor,
                        'optimizer': optimizer_clip.state_dict(),
                    }, checkpoint_clip)
                earlystop_cnt = 0
            else:
                print('Epoch %d: %s did not %s' % (epoch + 1, args.monitor, str_op))
                earlystop_cnt += 1

            scheduler.step()
            if cfgs['cfgs']['clip_train'] == True:
                scheduler_clip.step()

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
    model_clip = CLIP()
    model_clip = model_clip.cuda()
    test(test_loader, model, checkpoint, checkpoint_clip, lable_path, pred_path, model_clip)

def train(train_loader, model, criterion, criterion_clip, optimizer, epoch, model_clip, optimizer_clip=None):
    losses = AverageMeter()
    acces = AverageMeter()
    model.train()
    
    for i, (inputs, target) in enumerate(train_loader):
        #input:[bs,20,75]
        output, sekeleton_embeddings = model(inputs.cuda())  #sekeleton_embeddings:[bs, 512, 1, 20]
        #print(target.shape)#[bs]
        #print(inputs.shape)#[bs,20,75]
  
        #target:[bs]
        #target = target.cuda(async = True)
        if cfgs['cfgs']['network']=='SGN_CLIP':
            loss = criterion_clip(sekeleton_embeddings, target)
            target = target.cuda()
            acc = accuracy_clip(sekeleton_embeddings.data, target, model_clip)
            
        # measure accuracy and record loss
        if cfgs['cfgs']['network']=='SGN':
            target = target.cuda()
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

        #======================================================================
        # For Clip Text Encoder Fine-tuning
        #======================================================================
        losses_clip = AverageMeter()
        acces_clip = AverageMeter()
    
        if cfgs['cfgs']['clip_train'] == True:
            model_clip.train()
            
            _inputs, _targets = get_data_for_clip_finetuning(model_clip, train_loader)
            #_inputs:[bs,120,20,75]
            #_targets:[bs,120,512]
            
            bs = _inputs.shape[0]
            sekeleton_embeddings=torch.zeros((120,bs,20,512))
            for _i in range(0,120):
                output, sekeleton_embedding = model(_inputs[:,_i,:,:].cuda()) 
                sekeleton_embedding=sekeleton_embedding.view(bs,20,512)
                sekeleton_embeddings[_i,:,:,:] = sekeleton_embedding[:,:,:]
            
            sekeleton_embeddings = sekeleton_embeddings.view(bs,120,20,512)
            
            loss_clip = criterion_clip(sekeleton_embeddings, _targets)
            action_class=model_clip.module.action_classes()
            acc_clip = accuracy_clip_train(sekeleton_embeddings.data.to(_targets.device), _targets, action_class)
            
            losses_clip.update(loss_clip.item(), _inputs.size(0))
            acces_clip.update(acc_clip[0], _inputs.size(0))
            
            # backward
            optimizer_clip.zero_grad()  # clear gradients out before each mini-batch
            loss_clip.backward()
            optimizer_clip.step()

            if (i + 1) % args.print_freq == 0:
                print('clip-Epoch-{:<3d} {:3d} clip-batches\t'
                    'clip-loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'clip-accu {acc.val:.3f} ({acc.avg:.3f})'.format(
                    epoch + 1, i + 1, loss=losses_clip, acc=acces_clip))
        
    return losses.avg, acces.avg, losses_clip.avg, acces_clip.avg


def validate(val_loader, model, criterion, criterion_clip, model_clip):
    losses = AverageMeter()
    acces = AverageMeter()
    model.eval()

    for i, (inputs, target) in enumerate(val_loader):
        target = target.cuda()
        
        with torch.no_grad():
            output, sekeleton_embeddings = model(inputs.cuda())
        
        with torch.no_grad():
            if cfgs['cfgs']['network']=='SGN_CLIP':
                loss = criterion_clip(sekeleton_embeddings, target)
                acc = accuracy_clip(sekeleton_embeddings.data, target, model_clip)
            if cfgs['cfgs']['network']=='SGN':
                loss=criterion(output,target)
                acc = accuracy(output.data, target)
            
        # measure accuracy and record loss
        losses.update(loss.item(), inputs.size(0))
        acces.update(acc[0], inputs.size(0))
        
        #======================================================================
        # For Clip Text Encoder Fine-tuning
        #======================================================================
        losses_clip = AverageMeter()
        acces_clip = AverageMeter()
    
        if cfgs['cfgs']['clip_train'] == True:
            model_clip.eval()
            
            with torch.no_grad():
                _inputs, _targets = get_data_for_clip_finetuning(model_clip, val_loader)
                #_inputs:[bs,120,20,75]
                #_targets:[bs,120,512]
                
                bs = _inputs.shape[0]
                sekeleton_embeddings=torch.zeros((bs,120,20,512))
                for _i in range(0,120):
                    output, sekeleton_embedding = model(_inputs[:,_i,:,:].cuda()) 
                    sekeleton_embeddings[:,_i,:,:] = sekeleton_embedding.view(bs,20,512)
            
                loss_clip = criterion_clip(sekeleton_embeddings, _targets)
                action_class=model_clip.module.action_classes()
                acc_clip = accuracy_clip_train(sekeleton_embeddings.data.to(_targets.device), _targets, action_class)
                
            losses_clip.update(loss_clip.item(), _inputs.size(0))
            acces_clip.update(acc_clip[0], _inputs.size(0))

    return losses.avg, acces.avg,losses_clip.avg, acces_clip.avg


def test(test_loader, model, checkpoint, checkpoint_clip, lable_path, pred_path, model_clip):
    acces = AverageMeter()
    acces_clip = AverageMeter()
    # load learnt model that obtained best performance on validation set
    model.load_state_dict(torch.load(checkpoint)['state_dict'])
    model.eval()
    
    model_clip.load_state_dict(torch.load(checkpoint_clip)['state_dict'])
    model_clip.eval()

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
            acc = accuracy(output.data, target.cuda())
        if cfgs['cfgs']['network']=='SGN_CLIP':
            acc = accuracy_clip(skeleton_embedding.data, target, model_clip)
        acces.update(acc[0], inputs.size(0))

        #======================================================================
        # For Clip Text Encoder Fine-tuning
        #======================================================================
        if cfgs['cfgs']['clip_train']==True:
            with torch.no_grad():
                _inputs, _targets = get_data_for_clip_finetuning(model_clip, test_loader)
                #_inputs:[bs,120,20,75]
                #_targets:[bs,120,512]
                
                bs = _inputs.shape[0]
                sekeleton_embeddings=torch.zeros((bs,120,20,512))
                for _i in range(0,120):
                    output, sekeleton_embedding = model(_inputs[:,_i,:,:].cuda())
                    sekeleton_embeddings[:,_i,:,:] = sekeleton_embedding.view(bs,20,512)
                action_class=model_clip.module.action_classes()
                acc_clip = accuracy_clip_train(sekeleton_embeddings.data.to(_targets.device), _targets,action_class)
                acces_clip.update(acc_clip[0], _inputs.size(0))

    label_output = np.concatenate(label_output, axis=0)
    np.savetxt(lable_path, label_output, fmt='%d')
    pred_output = np.concatenate(pred_output, axis=0)
    np.savetxt(pred_path, pred_output, fmt='%f')

    print('Test: accuracy {:.3f}, time: {:.2f}s'
        .format(acces.avg, time.time() - t_start))
    
    if cfgs['cfgs']['clip_train']==True:
        print('Test (clip): accuracy {:.3f}, time: {:.2f}s'
            .format(acces_clip.avg, time.time() - t_start))


def accuracy(output, target):
    batch_size = target.size(0)
    _, pred = output.topk(1, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    correct = correct.view(-1).float().sum(0, keepdim=True)
    return correct.mul_(100.0 / batch_size)

def accuracy_clip(skeleton_embedding, target, model_clip):
    batch_size = target.size(0)
    clip_text_embeddings=model_clip.encode_text(model_clip.ntu120_text_tokens()).t() #[512,120]
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
    for i in range(0, batch_size):
        arr=[]
        skeleton_emb = skeleton_embedding[i,:]
        for k in range(0,120):
            target_text_emb = clip_text_embeddings[i,:,k]
            cosine_similarity=F.cosine_similarity(target_text_emb, skeleton_emb, dim=-1)
            arr.append(cosine_similarity)
            
        topk_val, topk_idx = torch.tensor(arr).topk(5, -1, True, True)
        if target[i].cpu() in topk_idx:
            cnt = cnt+1 
    
    correct=[1]
    correct[0]=100*(cnt/batch_size)
    acc=torch.tensor(correct)
    
    return acc

def accuracy_clip_train(skeleton_embeddings, targets, action_classes):
    bs=skeleton_embeddings.shape[0] 
    text_embeddings=targets #[bs,120,512]
    skeleton_embeddings=torch.mean(skeleton_embeddings,dim=2) #[bs,120,512]
    
    cnt=0
    target_text=""
    pred_text=""
    for k in range(0,bs):
        text_features = text_embeddings[k,:,:]/text_embeddings[k,:,:].norm(dim=-1, keepdim=True)
        skeleton_features = skeleton_embeddings[k,:,:]/skeleton_embeddings[k,:,:].norm(dim=-1, keepdim=True) #[120,512]
        
        similarity = (100.0 * (text_features.type(torch.DoubleTensor) @ skeleton_features.type(torch.DoubleTensor).t())).softmax(dim=-1)
     
        rnd_idx=random.randint(0,119)
        for i in range(0,120):
            _, topk_idx = torch.tensor(similarity).topk(5, -1, True, True)
            topk_idx=topk_idx.cpu().numpy()
            if i in topk_idx[0]:
                cnt+=1
            if i==rnd_idx:
                target_text=action_classes[int(i)]
                pred_text=action_classes[int(topk_idx[0][0])]
    print('target:',target_text,' / pred:',pred_text)
        
    correct=[1]
    correct[0]=100*(cnt/(bs*120))
    acc=torch.tensor(correct)
    
    return acc

def get_data_for_clip_finetuning(model_clip, train_loader):
    k=0
    b=0
    bs=cfgs['cfgs']['batch_size']
    inputs_list120=torch.zeros((bs,120,20,75))

    while b < bs:
        while k < 120:
            for _i, (_inputs, _target) in enumerate(train_loader):
                for j in range(0,bs):
                    if _target[j] == k:
                        inputs_list120[b,k,:,:]=_inputs[j,:,:]
                        k+=1
                        if k==120: break
                if k==120: break
        b+=1
    
    text_features=model_clip(model_clip.module.ntu120_text_tokens())
    text_features=text_features.unsqueeze(0).expand(bs,-1,-1)#[bs,120,512]
    
    _inputs=inputs_list120 #[bs,120,20,75]
    _targets=text_features #[bs,120,512]
    
    return _inputs, _targets
            
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
    def __init__(self, model_clip, type='cosine_similarity'):
        super(CLIPLoss, self).__init__()
        self.model_clip=model_clip
        self.ntu120_text_tokens=model_clip.ntu120_text_tokens()
        self.type=type
        
    def forward(self, skeleton_embeddings, target):
        text_embeddings=self.model_clip.encode_text(self.ntu120_text_tokens).cpu() #[120,512]
        text_features = text_embeddings[target.cpu().numpy(), :]
        text_features = torch.tensor(text_features).to(skeleton_embeddings.device)
            
        if self.type == 'cosine_similarity':
            skeleton_embeddings = skeleton_embeddings.view(-1,512) #[20*bs,512]
            cosine_similarity = F.cosine_similarity(text_features.unsqueeze(0), skeleton_embeddings.unsqueeze(1), dim=-1) #[20*bs,bs]
            clip_loss = 1-cosine_similarity.view(-1).mean()
            loss = clip_loss
        elif self.type=='cross_entropy':
            skeleton_embeddings = skeleton_embeddings.view(20,-1,512).mean(dim=0) #[bs,512]
            skeleton_embeddings_prob=F.softmax(skeleton_embeddings,dim=-1)
            cross_entropy=F.cross_entropy(skeleton_embeddings_prob,target.to(skeleton_embeddings_prob.device))
            loss = cross_entropy
        
        return loss

class CLIPTrainLoss(nn.Module):
    def __init__(self):
        super(CLIPTrainLoss, self).__init__()
    
    def forward(self, skeleton_embeddings, text_embeddings):
        # extract feature representations of each modality
        bs=skeleton_embeddings.shape[0]
        skeleton_embeddings = skeleton_embeddings.mean(dim=2) #[bs,120,20,512]
        text_features = text_embeddings/text_embeddings.norm(dim=-1, keepdim=True) #[bs,120,512]
        skeleton_features = skeleton_embeddings/skeleton_embeddings.norm(dim=-1, keepdim=True) #[bs,120,512]
        
        # cosine similarity as logits
        logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        logit_scale = logit_scale.exp()
        #logit_scale=1.0
        
        loss=0.0
        for i in range(0,bs):
            logits = logit_scale * (text_features[i,:,:].type(torch.DoubleTensor).to(skeleton_features.device) @ skeleton_features[i,:,:].type(torch.DoubleTensor).t())
            labels = torch.tensor(np.arange(120))
            loss += F.cross_entropy(logits,labels)
        
        loss/=bs
        
        return loss
    
        
if __name__ == '__main__':
    main()
    
