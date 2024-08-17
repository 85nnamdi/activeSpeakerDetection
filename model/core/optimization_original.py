import os
from re import X
import time
import copy
import torch
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

# Writer will output to ./runs/ directory by default
writer = SummaryWriter()

def optimize_av_losses(model, dataloader_train, data_loader_val, device,
                       criterion, optimizer, scheduler, num_epochs,
                       models_out=None, log=None):

    for epoch in range(num_epochs):
        print()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        outs_train = _train_model_av_losses(model, dataloader_train, optimizer,
                                            criterion, scheduler, device)
        outs_val = _test_model_av_losses(model, data_loader_val, optimizer,
                                         criterion, scheduler, device)

        train_loss, train_loss_a, train_loss_v, train_auc, train_ap = outs_train
        val_loss, val_loss_a, val_loss_v, val_auc, val_ap = outs_val

        if models_out is not None:
            model_target = os.path.join(models_out, str(epoch+1)+'.pth')
            print('save model to ', model_target)
            torch.save(model.state_dict(), model_target)

        if log is not None:
            log.writeDataLog([epoch+1, train_loss, train_auc, train_ap, val_loss, val_auc, val_ap])

    return model


def _train_model_av_losses(model, dataloader, optimizer, criterion,
                              scheduler, device):
    softmax_layer = torch.nn.Softmax(dim=1)
    model.train()
    pred_lst = []
    label_lst = []

    running_loss_av = 0.0
    running_loss_a = 0.0
    running_loss_v = 0.0
    running_corrects = 0

    # Iterate over data
    for idx, dl in enumerate(dataloader):
        print('\t Train iter ', idx, '/', len(dataloader), end='\r')
        data, av_label, audio_label = dl
        audio_data, video_data = data

        video_data = video_data.to(device)
        audio_data = audio_data.to(device)
        av_label = av_label.to(device)
        audio_label = audio_label.to(device)

        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            av_out, a_out, v_out, _ = model(audio_data, video_data)
            _, preds = torch.max(av_out, 1)

            loss_av = criterion(av_out, av_label)
            loss_a = criterion(a_out, audio_label)
            loss_v = criterion(v_out, av_label)
            loss = loss_av + loss_a + loss_v

            loss.backward()
            optimizer.step()

        with torch.set_grad_enabled(False):
            label_lst.extend(av_label.cpu().numpy().tolist())
            pred_lst.extend(softmax_layer(av_out).cpu().numpy()[:, 1].tolist())

        # statistics
        running_loss_av += loss_av.item()
        running_loss_a += loss_a.item()
        running_loss_v += loss_v.item()
        running_corrects += torch.sum(preds == av_label.data)

    scheduler.step()


    epoch_loss_av = running_loss_av / len(dataloader)
    epoch_loss_a = running_loss_a / len(dataloader)
    epoch_loss_v = running_loss_v / len(dataloader)
    epoch_acc = running_corrects.double() / len(dataloader)

    epoch_auc = roc_auc_score(label_lst, pred_lst)
    epoch_ap = average_precision_score(label_lst, pred_lst)
    print('AV Loss: {:.4f} A Loss: {:.4f}  V Loss: {:.4f}  AUC: {:.4f} AP: {:.4f}'.format(
          epoch_loss_av, epoch_loss_a, epoch_loss_v, epoch_auc, epoch_ap))
    return epoch_loss_av, epoch_loss_a, epoch_loss_v, epoch_auc, epoch_ap


def _test_model_av_losses(model, dataloader, optimizer, criterion, scheduler, device):
    softmax_layer = torch.nn.Softmax(dim=1)

    model.eval()   # Set model to evaluate mode
    pred_lst = []
    label_lst = []

    running_loss_av = 0.0
    running_loss_a = 0.0
    running_loss_v = 0.0
    running_corrects = 0

    # Iterate over data.
    for idx, dl in enumerate(dataloader):
        print('\t Val iter ', idx, '/', len(dataloader), end='\r')
        data, av_label, audio_label = dl
        audio_data, video_data = data
        video_data = video_data.to(device)
        audio_data = audio_data.to(device)
        av_label = av_label.to(device)
        audio_label = audio_label.to(device)

        # forward
        with torch.set_grad_enabled(False):
            av_out, a_out, v_out, _ = model(audio_data, video_data)
            _, preds = torch.max(av_out, 1)
            loss_av = criterion(av_out, av_label)
            loss_a = criterion(a_out, audio_label)
            loss_v = criterion(v_out, av_label)

            label_lst.extend(av_label.cpu().numpy().tolist())
            pred_lst.extend(softmax_layer(av_out).cpu().numpy()[:, 1].tolist())

        # statistics
        running_loss_av += loss_av.item()
        running_loss_a += loss_a.item()
        running_loss_v += loss_v.item()
        running_corrects += torch.sum(preds == av_label.data)

    epoch_loss_av = running_loss_av / len(dataloader)
    epoch_loss_a = running_loss_a / len(dataloader)
    epoch_loss_v = running_loss_v / len(dataloader)
    epoch_acc = running_corrects.double() / len(dataloader)

    epoch_auc = roc_auc_score(label_lst, pred_lst)
    epoch_ap = average_precision_score(label_lst, pred_lst)
    print('AV Loss: {:.4f}  A Loss: {:.4f}  V Loss: {:.4f} Acc: {:.4f} auROC: {:.4f} AP: {:.4f}'.format(
          epoch_loss_av, epoch_loss_a, epoch_loss_v, epoch_acc, epoch_auc, epoch_ap))

    return epoch_loss_av, epoch_loss_a, epoch_loss_v, epoch_auc, epoch_ap


def optimize_asc(model, dataloader_train, data_loader_val, device,
                   criterion, optimizer, scheduler, num_epochs,
                   models_out=None, log=None):

    for epoch in range(num_epochs):
        print()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        model, loss,train_auc = _train_model_asc(model, dataloader_train, optimizer,
                                             criterion, scheduler, device, epoch)

        val_loss, val_acc, val_auc, val_ap = _test_model_asc(model, data_loader_val,
                                                   optimizer, criterion,
                                                   scheduler, device, epoch)

        best_model_wts = copy.deepcopy(model.state_dict())
        model.load_state_dict(best_model_wts)

        if models_out is not None:
            model_target = os.path.join(models_out, str(epoch+1)+'pose.pth')
            print('save model to ', model_target)
            torch.save(model.state_dict(), model_target)

        if log is not None:
            log.writeDataLog([epoch+1, loss,train_auc, val_loss, val_acc, val_auc, val_ap])

    return model


def _train_model_asc(model, dataloader, optimizer, criterion, scheduler,
                           device, epoch_idx):
    scheduler.step()
    model.train()

    running_loss = 0.0
    running_corrects = 0

    # Iterate over data
    count_num_entries =0
    for idx, dl in enumerate(dataloader):
        print('\t Train iter ', idx, '/', len(dataloader), end='\r')

        feat_data, labels = dl
        
        feat_data = feat_data.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            outputs = model(feat_data)
            count_num_entries+=labels.size(0)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

        # statistics
        current_loss = loss.item()
        current_corrects = torch.sum(preds == labels.data)
        running_loss += current_loss
        running_corrects += current_corrects
        
        # Tensorboard
        writer.add_scalar('Train/Batch/Loss', current_loss,epoch_idx*len(dataloader)+idx)
        writer.add_scalar('Train/Batch/Accuracy', current_corrects/(labels.size(0)),epoch_idx*len(dataloader)+idx)
        

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = running_corrects.double() / count_num_entries #len(dataloader)
    writer.add_scalar('Train/Epoch/Loss', epoch_loss,epoch_idx)
    writer.add_scalar('Train/Epoch/Accuracy', epoch_acc, epoch_idx)

    print('train Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

    return model, epoch_loss,epoch_acc


def _test_model_asc(model, dataloader, optimizer, criterion, scheduler,
                          device, epoch_idx):
    softmax_layer = torch.nn.Softmax()

    model.eval()   # Set model to evaluate mode
    pred_lst = []
    label_lst = []

    running_loss = 0.0
    running_corrects = 0

    # Iterate over data.
    count_num_entries =0
    for idx, dl in enumerate(dataloader):
        print('\t Val iter ', idx, '/', len(dataloader), end='\r')
        feat_data, labels = dl
        feat_data = feat_data.to(device)
        labels = labels.to(device)

        # forward
        with torch.set_grad_enabled(False):
            outputs = model(feat_data)
            count_num_entries+=labels.size(0)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            label_lst.extend(labels.cpu().numpy().tolist())
            pred_lst.extend(softmax_layer(outputs).cpu().numpy()[:, 1].tolist())

        # statistics
        running_loss += loss.item()
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = running_corrects.double() / count_num_entries #len(dataloader)

    epoch_auc = roc_auc_score(label_lst, pred_lst)
    epoch_ap = average_precision_score(label_lst, pred_lst)
    
    print('Val Loss: {:.4f} Acc: {:.4f} auROC: {:.4f} AP: {:.4f}'.format(
          epoch_loss, epoch_acc, epoch_auc, epoch_ap))
    # Tensorboard
    writer.add_scalar('Val/Epoch/Loss', epoch_loss,epoch_idx)
    writer.add_scalar('Val/Epoch/Accuracy', epoch_acc, epoch_idx)
    writer.add_scalar('Val/Epoch/AUC', epoch_auc,epoch_idx)
    writer.add_scalar('Val/Epoch/mAP', epoch_ap, epoch_idx)
    return epoch_loss,epoch_acc, epoch_auc, epoch_ap
