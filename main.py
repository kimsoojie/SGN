# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import argparse
import time
import shutil
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import os.path as osp
import csv
import numpy as np

np.random.seed(1337)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
from model import SGN
from model_fc import ActionText
import clip
from data import NTUDataLoaders, AverageMeter
import fit
from util import make_dir, get_num_classes
from label_text import text, text_sysu, text_nucla
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='Skeleton-Based Action Recgnition')
fit.add_fit_args(parser)
parser.set_defaults(
    network='SGN',
    dataset = 'NTU120',
    dataloader_type='NUCLA',
    case = 0,
    batch_size=  10,
    max_epochs=120,
    monitor='val_acc',
    lr=0.001,
    weight_decay=0.0001,
    lr_factor=0.1,
    workers=16,
    print_freq = 20,
    train = 0,
    seg = 20,
    )
args = parser.parse_args()



def load_and_freeze_clip(clip_version):
    clip_model, clip_preprocess = clip.load(clip_version, device='cpu',
                                            jit=False)  # Must set jit=False for training
    clip.model.convert_weights(
        clip_model)  # Actually this line is unnecessary since clip by default already on float16

    # Freeze CLIP weights
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False

    return clip_model
    
def encoded_text(clip_model, text):
    text_token = tokenize(text)
    enc_text = clip_model.encode_text(text_token).float()
    return enc_text

def encoded_text_normalized(clip_model, text):
    enc_text = encoded_text(clip_model, text)
    normalized_vector = F.normalize(enc_text, p=2, dim=1)
    return normalized_vector

def tokenize(raw_text, device="cuda"):
    max_text_len = 20

    default_context_length = 77
    context_length = max_text_len + 2 # start_token + 20 + end_token
    assert context_length < default_context_length
    texts = clip.tokenize(raw_text, context_length=context_length, truncate=True).to(device) # [bs, context_length] # if n_tokens > context_length -> will truncate
    # print('texts', texts.shape)
    zero_pad = torch.zeros([texts.shape[0], default_context_length-context_length], dtype=texts.dtype, device=texts.device)
    texts = torch.cat([texts, zero_pad], dim=1)
    return texts


clip_model = load_and_freeze_clip("ViT-B/32")
clip_model = clip_model.cuda()
text_embed = encoded_text(clip_model, text)
text_embed_sysu = encoded_text(clip_model, text_sysu)
text_embed_nucla = encoded_text(clip_model, text_nucla)

def main():

    args.num_classes = get_num_classes(args.dataset)
    model = SGN(args.num_classes, args.dataset, args.seg, args)
    model_fc = ActionText()
    total = get_n_params(model)
    print(model)
    print('The number of parameters: ', total)
    print('The modes is:', args.network)

    if torch.cuda.is_available():
        print('It is using GPU!')
        model = model.cuda()
        model_fc = model_fc.cuda()

    criterion = LabelSmoothingLoss(args.num_classes, smoothing=0.1).cuda()
    optimizer = optim.Adam(
        model_fc.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay
    )

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
    
    #ntu_loaders = NTUDataLoaders('SYSU', args.case, seg=args.seg)
    ntu_loaders = NTUDataLoaders(args.dataloader_type, args.case, seg=args.seg)
    train_loader = ntu_loaders.get_train_loader(args.batch_size, args.workers)
    val_loader = ntu_loaders.get_val_loader(args.batch_size, args.workers)
    train_size = ntu_loaders.get_train_size()
    val_size = ntu_loaders.get_val_size()
    test_loader = ntu_loaders.get_test_loader(32, args.workers)

    #print('Train on %d samples, validate on %d samples' % (train_size, val_size))
    print(len(train_loader))
    print(len(val_loader))
    print(len(test_loader))
    
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
    model.load_state_dict(torch.load(checkpoint)['state_dict'])
    if args.train ==1:
        for epoch in range(args.start_epoch, args.max_epochs):

            print(epoch, optimizer.param_groups[0]['lr'])

            t_start = time.time()
            train_loss, train_acc = train(train_loader, model, model_fc, criterion, optimizer, epoch, args.dataset)
            val_loss, val_acc = validate(val_loader, model, model_fc, criterion,args.dataset)
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
                    'state_dict': model_fc.state_dict(),
                    'best': best,
                    'monitor': args.monitor,
                    'optimizer': optimizer.state_dict(),
                }, checkpoint.replace(".pth", "_fc.pth"))
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
    model_fc = ActionText()
    model_fc = model_fc.cuda()
    test(test_loader, model, model_fc, checkpoint, lable_path, pred_path, args.dataset)


def train(train_loader, model, model_fc, criterion, optimizer, epoch, dataset):
    losses = AverageMeter()
    acces = AverageMeter()
    model.eval()
    model_fc.train()

    for i, (inputs, target) in enumerate(train_loader):
        if args.dataset == 'NTU120' and (args.dataloader_type == 'SYSU' or args.dataloader_type == 'NUCLA'):
                target_shape = [inputs.shape[0],inputs.shape[1],75]
                new_inputs = np.zeros(target_shape)
                new_inputs[:inputs.shape[0], :inputs.shape[1], :inputs.shape[2]] = inputs
                inputs = torch.FloatTensor(new_inputs)
                
        _, action_features = model(inputs.cuda())
        output = model_fc(action_features.detach())
        if dataset == 'SYSU':
            cosine_sim = torch.cosine_similarity(output.unsqueeze(1), text_embed_sysu.unsqueeze(0), dim=2)
        elif dataset == 'NTU120':
            cosine_sim = torch.cosine_similarity(output.unsqueeze(1), text_embed.unsqueeze(0), dim=2)
        elif dataset == 'NUCLA':
            cosine_sim = torch.cosine_similarity(output.unsqueeze(1), text_embed_nucla.unsqueeze(0), dim=2)
            
        target = target.cuda()

        loss = criterion(cosine_sim, target)
        
        # measure accuracy and record loss
        acc = accuracy(cosine_sim.data, target)
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


def validate(val_loader, model, model_fc, criterion, dataset):
    losses = AverageMeter()
    acces = AverageMeter()
    model.eval()
    model_fc.eval()

    for i, (inputs, target) in enumerate(val_loader):
        with torch.no_grad():
            if args.dataset == 'NTU120' and (args.dataloader_type == 'SYSU' or args.dataloader_type == 'NUCLA'):
                target_shape = [inputs.shape[0],inputs.shape[1],75]
                new_inputs = np.zeros(target_shape)
                new_inputs[:inputs.shape[0], :inputs.shape[1], :inputs.shape[2]] = inputs
                inputs = torch.FloatTensor(new_inputs)
                
            output, action_features = model(inputs.cuda())
            output = model_fc(action_features.detach())
            if dataset == 'SYSU':
                cosine_sim = torch.cosine_similarity(output.unsqueeze(1), text_embed_sysu.unsqueeze(0), dim=2)
            elif dataset == 'NTU120':
                cosine_sim = torch.cosine_similarity(output.unsqueeze(1), text_embed.unsqueeze(0), dim=2)
            elif dataset == 'NUCLA':
                cosine_sim = torch.cosine_similarity(output.unsqueeze(1), text_embed_nucla.unsqueeze(0), dim=2)
        target = target.cuda()
        with torch.no_grad():
            loss = criterion(cosine_sim, target)

        # measure accuracy and record loss
        acc = accuracy(cosine_sim.data, target)
        losses.update(loss.item(), inputs.size(0))
        acces.update(acc[0], inputs.size(0))

    return losses.avg, acces.avg


def test(test_loader, model, model_fc, checkpoint, lable_path, pred_path, dataset):
    acces = AverageMeter()
    acces2 = AverageMeter()
    # load learnt model that obtained best performance on validation set
    # model.load_state_dict(torch.load(checkpoint.replace(".pth", "_new.pth"))['state_dict'])
    model.load_state_dict(torch.load(checkpoint)['state_dict'])
    model.eval()
    # model_fc.load_state_dict(torch.load(checkpoint.replace(".pth", "_new_fc.pth"))['state_dict'])
    model_fc.load_state_dict(torch.load(checkpoint.replace(".pth", "_fc.pth"))['state_dict'])
    model_fc.eval()

    label_output = list()
    pred_output = list()

    label=[]
    predicted=[]
    
    t_start = time.time()
    for i, (inputs, target) in enumerate(test_loader):
        with torch.no_grad():
            #print(inputs.shape)#torch.Size([160, 20, 60]):sysu/[160,20,75]:ntu120
            #print(target.shape)#torch.Size([32]):sysu
            
            if args.dataset == 'NTU120' and (args.dataloader_type == 'SYSU' or args.dataloader_type == 'NUCLA'):
                target_shape = [inputs.shape[0],inputs.shape[1],75]
                new_inputs = np.zeros(target_shape)
                new_inputs[:inputs.shape[0], :inputs.shape[1], :inputs.shape[2]] = inputs
                inputs = torch.FloatTensor(new_inputs)
            
            output2, action_features = model(inputs.cuda())
            output = model_fc(action_features)
            if args.dataloader_type == 'SYSU':
                cosine_sim = torch.cosine_similarity(output.unsqueeze(1), text_embed_sysu.unsqueeze(0), dim=2)
            elif args.dataloader_type == 'NUCLA':
                cosine_sim = torch.cosine_similarity(output.unsqueeze(1), text_embed_nucla.unsqueeze(0), dim=2)
            else:
                cosine_sim = torch.cosine_similarity(output.unsqueeze(1), text_embed.unsqueeze(0), dim=2)
            cosine_sim = cosine_sim.view((-1, inputs.size(0)//target.size(0), cosine_sim.size(1)))
            cosine_sim = cosine_sim.mean(1)          

        label_output.append(target.cpu().numpy())
        #pred_output.append(output2.cpu().numpy())
        pred_output.append(cosine_sim.cpu().numpy())

        acc = accuracy(cosine_sim.data, target.cuda())
        #acc = accuracy(output2.data, target.cuda())
        acces.update(acc[0], inputs.size(0))
        acc2 = accuracy(cosine_sim.data, target.cuda())
        acces2.update(acc2[0], inputs.size(0))
        
        
        _, pred = cosine_sim.data.topk(1, 1, True, True)
        label.append(target.cpu().numpy())
        predicted.append(pred.t().cpu().numpy()[0])
       
        
    if args.dataloader_type=='SYSU' or args.dataloader_type=='NUCLA' or args.dataloader_type=='NTU120':
        label=np.concatenate(label)
        predicted=np.concatenate(predicted)
        cm = confusion_matrix(label,predicted)
        total = np.sum(cm)
        cm = (cm/total)*100.
        print('total',total)
        
        if args.dataloader_type=='SYSU':
            #n=12
            n=text_sysu
            plt.figure(figsize=(20, 18))
        elif args.dataloader_type=='NUCLA':
            #n=10
            n=text_nucla
            plt.figure(figsize=(20, 18))
        elif args.dataloader_type=='NTU120':
            #n=120
            n=text
            plt.figure(figsize=(60, 60))
                
        
        h = sns.heatmap(cm, annot=True,  fmt=".1f",cmap='Greys', xticklabels=n, yticklabels=n)
        h.set_facecolor('white')
    
        #t=0
        #for text_x, text_y in zip(h.get_xticklabels(),h.get_yticklabels()):
        #    if t%2 == 0:
        #        text_x.set(rotation=90, ha='center', va='center',y=-0.03)
        #        text_y.set(rotation=0, ha='center', va='center', x=-0.03)
        #    t += 1
            
       
        for t in h.texts:
            t.set(rotation=45, ha='center', va='center')
            
            
        plt.xlabel("predicted label",)
        plt.ylabel("target label")
    
        plt.title(args.dataloader_type+" (SGN+CLIP)")
        plt.savefig('cm_'+args.dataloader_type+'.jpg')
        plt.close()
        acc_224 = accuracy_score(label,predicted) * 100.
        print('**test accuracy:', acc_224)
    
    label_output = np.concatenate(label_output, axis=0)
    np.savetxt(lable_path, label_output, fmt='%d')
    pred_output = np.concatenate(pred_output, axis=0)
    np.savetxt(pred_path, pred_output, fmt='%f')

    print('Test: accuracy {:.3f}, time: {:.2f}s'
          .format(acces.avg, time.time() - t_start))
    print('Test (New): accuracy {:.3f}, time: {:.2f}s'
          .format(acces2.avg, time.time() - t_start))

def accuracy(output, target):
    batch_size = target.size(0)
    _, pred = output.topk(1, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    correct = correct.view(-1).float().sum(0, keepdim=True)
  
    #for i in range(0,batch_size):
    #    print(text[pred[0,i]],'/',text[target[i]])
    return correct.mul_(100.0 / batch_size)

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




if __name__ == '__main__':
    main()
    
