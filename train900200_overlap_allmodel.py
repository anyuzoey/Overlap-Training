#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gc
gc.collect()
import sys
from functions import *
# import cmapy
from pytorchtools import EarlyStopping
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Only GPU 1 is visible to this code

seeds = [5193,236]
# in order to get reproducable results
for seed in seeds:
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


    time1 = time.time()


    # In[2]:


    seismic_path = 'seistrain.npy'
    label_path = 'faulttrain.npy'


    t_start = time.time()
    seismic = np.load(seismic_path)
    fault = np.load(label_path)
    print("load in {} sec".format(time.time()-t_start))
    print(seismic.shape, fault.shape)
    IL, Z, XL = fault.shape

    modelname = "rcf"
    lossweight = 0.5
    startOverlapLossEpochNo = 10
    best_model_fpath ='{}_LRoverlap_0.5Loss12_{}Loss3_addafterepoch{}_seed{}.model'.format(modelname,lossweight,startOverlapLossEpochNo,seed)
    print(best_model_fpath)
    overlap = True
    best_iou_threshold=0.5
    epoches = 50
    patience = 10
    im_height = Z
    im_width = XL
    splitsize = 96
    stepsize = 48
    overlapsize = splitsize-stepsize
    pixelThre = int(0.03*splitsize*splitsize)
    print(pixelThre)


    horizontal_splits_number = int(np.ceil((im_width-overlapsize)/stepsize))
    print("horizontal_splits_number", horizontal_splits_number)
    width_after_pad = stepsize*horizontal_splits_number+overlapsize
    print("width_after_pad", width_after_pad)
    left_pad = int((width_after_pad-im_width)/2)
    right_pad = width_after_pad-im_width-left_pad
    print("left_pad,right_pad",left_pad,right_pad)

    vertical_splits_number = int(np.ceil((im_height-overlapsize)/stepsize))
    print("vertical_splits_number",vertical_splits_number)
    height_after_pad = stepsize*vertical_splits_number+overlapsize
    print("height_after_pad",height_after_pad)
    top_pad = int((height_after_pad-im_height)/2)
    bottom_pad = height_after_pad-im_height-top_pad
    print("top_pad,bottom_pad", top_pad,bottom_pad)


    # In[17]:

    # training set --------------
    t_start = time.time()
    X1 = []
    Y1 = []
    X2 = []
    Y2 = []
    deletenumbers = 0
    for i in range(900):
        mask = fault[i]
        splits = split_Image(mask, True,top_pad,bottom_pad,left_pad,right_pad,splitsize,stepsize,vertical_splits_number,horizontal_splits_number)
        t = (splits.sum((1,2)) >= pixelThre)
        label_element_index1 = list(compress(range(len(t)), t))
        label_element_index2 = [x+1 for x in label_element_index1]
        Y1.extend(splits[label_element_index1])
        Y2.extend(splits[label_element_index2]) 

        img = seismic[i]
        splits = split_Image(img, True,top_pad,bottom_pad,left_pad,right_pad,splitsize,stepsize,vertical_splits_number,horizontal_splits_number)
        X1.extend(splits[label_element_index1])
        X2.extend(splits[label_element_index2])

    print(deletenumbers)
    print(len(Y1))
    print(len(X1))
    print(len(Y2))
    print(len(X2))
    print(X1[0].shape)
    print("read images in {} sec".format(time.time()-t_start))

    t_start = time.time()
    X1 = np.asarray(X1, dtype=np.float32)
    Y1 = np.asarray(Y1, dtype=np.float32)
    X2 = np.asarray(X2, dtype=np.float32)
    Y2 = np.asarray(Y2, dtype=np.float32)
    print("X1.shape", X1.shape)
    print("Y1.shape",Y1.shape)
    print("X2.shape",X2.shape)
    print("Y2.shape",Y2.shape)
    print("read images in {} sec".format(time.time()-t_start))


    if len(Y1.shape) == 3:
        Y1 = np.expand_dims(Y1, axis=-1)
    if len(X1.shape) == 3:
        X1 = np.expand_dims(X1, axis=-1)
    if len(Y2.shape) == 3:
        Y2 = np.expand_dims(Y2, axis=-1)
    if len(X2.shape) == 3:
        X2 = np.expand_dims(X2, axis=-1)
    print("X1.shape", X1.shape)
    print("Y1.shape",Y1.shape)
    print("X2.shape",X2.shape)
    print("Y2.shape",Y2.shape)


    X_train1 = X1
    Y_train1 = Y1
    X_train2 = X2
    Y_train2 = Y2


    # validation set --------------
    t_start = time.time()
    X1 = []
    Y1 = []
    X2 = []
    Y2 = []
    deletenumbers = 0
    for i in range(900,1100,1):
        mask = fault[i]
        splits = split_Image(mask, True,top_pad,bottom_pad,left_pad,right_pad,splitsize,stepsize,vertical_splits_number,horizontal_splits_number)
        t = (splits.sum((1,2)) >= pixelThre)
        label_element_index1 = list(compress(range(len(t)), t))
        label_element_index2 = [x+1 for x in label_element_index1]
        Y1.extend(splits[label_element_index1])
        Y2.extend(splits[label_element_index2]) 

        img = seismic[i]
        splits = split_Image(img, True,top_pad,bottom_pad,left_pad,right_pad,splitsize,stepsize,vertical_splits_number,horizontal_splits_number)
        X1.extend(splits[label_element_index1])
        X2.extend(splits[label_element_index2])

    print(deletenumbers)
    print(len(Y1))
    print(len(X1))
    print(len(Y2))
    print(len(X2))
    print(X1[0].shape)
    print("read images in {} sec".format(time.time()-t_start))

    t_start = time.time()
    X1 = np.asarray(X1, dtype=np.float32)
    Y1 = np.asarray(Y1, dtype=np.float32)
    X2 = np.asarray(X2, dtype=np.float32)
    Y2 = np.asarray(Y2, dtype=np.float32)
    print("X1.shape", X1.shape)
    print("Y1.shape",Y1.shape)
    print("X2.shape",X2.shape)
    print("Y2.shape",Y2.shape)
    print("read images in {} sec".format(time.time()-t_start))


    if len(Y1.shape) == 3:
        Y1 = np.expand_dims(Y1, axis=-1)
    if len(X1.shape) == 3:
        X1 = np.expand_dims(X1, axis=-1)
    if len(Y2.shape) == 3:
        Y2 = np.expand_dims(Y2, axis=-1)
    if len(X2.shape) == 3:
        X2 = np.expand_dims(X2, axis=-1)
    print("X1.shape", X1.shape)
    print("Y1.shape",Y1.shape)
    print("X2.shape",X2.shape)
    print("Y2.shape",Y2.shape)


    X_val1 = X1
    Y_val1 = Y1
    X_val2 = X2
    Y_val2 = Y2

    print("X_train1",X_train1.shape)
    print("X_val1",X_val1.shape)
    print("Y_train1",Y_train1.shape)
    print("Y_val1",Y_val1.shape)

    print("X_train2",X_train2.shape)
    print("X_val2",X_val2.shape)
    print("Y_train2",Y_train2.shape)
    print("Y_val2",Y_val2.shape)

    # In[ ]:


    # t_start = time.time()
    # origin_train_size = len(X_train)
    # print(origin_train_size)
    # X_train_aug = np.zeros((origin_train_size*1,splitsize,splitsize,1))
    # Y_train_aug = np.zeros((origin_train_size*1,splitsize,splitsize,1))
    # for i in range(len(X_train)):
    #     for j in range(1):
    #         aug = strong_aug(p=1)
    #         augmented = aug(image=X_train[i], mask=Y_train[i])
    #         X_train_aug[origin_train_size*j + i] = augmented['image']
    #         Y_train_aug[origin_train_size*j + i] = augmented['mask']
    # print("read images in {} sec".format(time.time()-t_start))

    # if len(X_train)==origin_train_size:
    #     X_train = np.append(X_train,X_train_aug, axis=0)
    # if len(Y_train)==origin_train_size:
    #     Y_train = np.append(Y_train, Y_train_aug, axis=0)
    # print("X_train after aug",X_train.shape) 
    # print("Y_train after aug",Y_train.shape)
    # print("read images in {} sec".format(time.time()-t_start))

    X_train1 = X_train1.astype(np.float32)
    Y_train1 = Y_train1.astype(np.float32)
    X_train1 = np.moveaxis(X_train1,-1,1)
    Y_train1 = np.moveaxis(Y_train1,-1,1)
    X_val1 = np.moveaxis(X_val1,-1,1)
    Y_val1 = np.moveaxis(Y_val1,-1,1)

    X_train2 = X_train2.astype(np.float32)
    Y_train2 = Y_train2.astype(np.float32)
    X_train2 = np.moveaxis(X_train2,-1,1)
    Y_train2 = np.moveaxis(Y_train2,-1,1)
    X_val2 = np.moveaxis(X_val2,-1,1)
    Y_val2 = np.moveaxis(Y_val2,-1,1)

    print("X_train1",X_train1.shape)
    print("X_val1",X_val1.shape)
    print("Y_train1",Y_train1.shape)
    print("Y_val1",Y_val1.shape)

    print("X_train2",X_train2.shape)
    print("X_val2",X_val2.shape)
    print("Y_train2",Y_train2.shape)
    print("Y_val2",Y_val2.shape)


    # In[6]:


    # idea from: https://www.kaggle.com/erikistre/pytorch-basic-u-net
    class faultsDataset(torch.utils.data.Dataset):

        def __init__(self, seismic1, seismic2, fault1, fault2, train):
            """
            Args:
                text_file(string): path to text file
                root_dir(string): directory with all train images
            """
            self.train = train
            self.image1 = seismic1
            self.image2 = seismic2
            self.mask1 = fault1
            self.mask2 = fault2

        def __len__(self):
            return len(self.image1)

        def __getitem__(self, idx):
            image1 = self.image1[idx]
            image2 = self.image2[idx]
            mask1 = self.mask1[idx]
            mask2 = self.mask2[idx]

            return (image1, image2, mask1, mask2)


    # In[32]:


    faults_dataset_train = faultsDataset(X_train1,X_train2,Y_train1,Y_train2, train=True)
    faults_dataset_val = faultsDataset(X_val1,X_val2,Y_val1,Y_val2, train=False)
    
    batch_size = 64 

    train_loader = torch.utils.data.DataLoader(dataset=faults_dataset_train, 
                                               batch_size=batch_size, 
                                               shuffle=True)

    val_loader = torch.utils.data.DataLoader(dataset=faults_dataset_val, 
                                               batch_size=batch_size, 
                                               shuffle=True)


    device = torch.device("cuda")
    if modelname =="unet":
        from model_zoo.UNET import Unet
        model = Unet()
        print("use model Unet")
    elif modelname == "deeplab":
        from model_zoo.DEEPLAB.deeplab import DeepLab
        model = DeepLab(backbone='mobilenet', num_classes=1, output_stride=16)
        print("use model DeepLab")               
    elif modelname == "rcf":
        from model_zoo.RCF import RCF
        model = RCF()
        print("use model RCF")        
    model = nn.DataParallel(model)
    model.to(device)
    summary(model, (1, 96, 96))


    optimizer = torch.optim.SGD(model.parameters(), lr=1e-6, momentum=0.9, weight_decay=0.0002)
    print("optimizer = torch.optim.SGD(model.parameters(), lr=1e-7, momentum=0.9, weight_decay=0.0002)")
    if modelname =="unet" or modelname == "deeplab":
        print("optimizer = torch.optim.Adam(model.parameters(), lr=0.01)")
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    elif modelname == "rcf":
        print("optimizer = torch.optim.SGD(model.parameters(), lr=1e-6, momentum=0.9, weight_decay=0.0002)")
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-6, momentum=0.9, weight_decay=0.0002)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',factor=0.1, patience=5, verbose=True)


    # In[ ]:



    # In[ ]:
#     mseloss = nn.MSELoss()
    bceloss = nn.BCELoss()
    logmaeloss = logMAEloss() # in function file
    mean_train_losses = []
    mean_val_losses = []
    mean_train_loss1 = []
    mean_train_loss2 = []
    mean_train_loss3 = []
    mean_val_loss1 = []
    mean_val_loss2 = []
    mean_val_loss3 = []
    mean_train_accuracies = []
    mean_val_accuracies = []
    train_losses_to_csv = []
    test_losses_to_csv = []
    t_start = time.time()
    early_stopping = EarlyStopping(patience=patience, verbose=True, delta = 0)
    for epoch in range(epoches):

        train_losses = []
        val_losses = []
        train_loss1 = []
        train_loss2 = []
        train_loss3 = []
        val_loss1 = []
        val_loss2 = []
        val_loss3 = []
        train_accuracies = []
        val_accuracies = []

        model.train()
        for image1, image2, mask1, mask2 in train_loader:

            image1 = Variable(image1.cuda())
            mask1 = Variable(mask1.cuda())
            output1 = model(image1)
            image2 = Variable(image2.cuda())
            mask2 = Variable(mask2.cuda())
            output2 = model(image2)

            loss = torch.zeros(1).cuda()
            loss1 = torch.zeros(1).cuda()
            loss2 = torch.zeros(1).cuda()
            loss3 = torch.zeros(1).cuda()

            y_preds = output1
            if modelname =="unet" or modelname == "deeplab":
                loss1 = bceloss(output1, mask1)
                loss2 = bceloss(output2, mask2)
                loss3 = logmaeloss(output1[:,:,:,stepsize:], output2[:,:,:,:stepsize].detach()) + logmaeloss(output2[:,:,:,:stepsize], output1[:,:,:,stepsize:].detach())
            elif modelname == "rcf":
                for o in output1:
                    loss1 = loss1 + cross_entropy_loss_RCF(o, mask1)
                for o in output2:
                    loss2 = loss2 + cross_entropy_loss_RCF(o, mask2)
                loss3 = loss3 + logmaeloss(output1[-1][:,:,:,stepsize:], output2[-1][:,:,:,:stepsize].detach(),reduction = 'sum') + logmaeloss(output2[-1][:,:,:,:stepsize], output1[-1][:,:,:,stepsize:].detach(),reduction = 'sum')
                y_preds = output1[-1]

            if overlap == False:
                loss = 0.5*(loss1+loss2)
            else:
                if epoch >=startOverlapLossEpochNo:
                    loss = 0.5*(loss1+loss2)+lossweight*loss3
                else:
                    loss = 0.5*(loss1+loss2)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_losses.append(loss.data)
            train_loss1.append(loss1.data)
            train_loss2.append(loss2.data)
            train_loss3.append(loss3.data)
            predicted_mask1 = y_preds > best_iou_threshold
            train_acc = iou_pytorch(predicted_mask1.squeeze(1).byte(), mask1.squeeze(1).byte())
            train_accuracies.append(train_acc.mean())

        model.eval()
        for image1, image2, mask1, mask2 in val_loader:
            image1 = Variable(image1.cuda())
            mask1 = Variable(mask1.cuda())
            output1 = model(image1)
            image2 = Variable(image2.cuda())
            mask2 = Variable(mask2.cuda())
            output2 = model(image2)

            loss = torch.zeros(1).cuda()
            loss1 = torch.zeros(1).cuda()
            loss2 = torch.zeros(1).cuda()
            loss3 = torch.zeros(1).cuda()
            y_preds = output1
            if modelname =="unet" or modelname == "deeplab":
                loss1 = bceloss(output1, mask1)
                loss2 = bceloss(output2, mask2)
                loss3 = logmaeloss(output1[:,:,:,stepsize:], output2[:,:,:,:stepsize].detach()) + logmaeloss(output2[:,:,:,:stepsize], output1[:,:,:,stepsize:].detach())
            elif modelname == "rcf":
                for o in output1:
                    loss1 = loss1 + cross_entropy_loss_RCF(o, mask1)
                for o in output2:
                    loss2 = loss2 + cross_entropy_loss_RCF(o, mask2)
                loss3 = loss3 + logmaeloss(output1[-1][:,:,:,stepsize:], output2[-1][:,:,:,:stepsize].detach(),reduction = 'sum') + logmaeloss(output2[-1][:,:,:,:stepsize], output1[-1][:,:,:,stepsize:].detach(),reduction = 'sum')
                y_preds = output1[-1]

            if overlap == False:
                loss = 0.5*(loss1+loss2)
            else:
                if epoch >=startOverlapLossEpochNo:
                    loss = 0.5*(loss1+loss2)+lossweight*loss3
                else:
                    loss = 0.5*(loss1+loss2)

            val_losses.append(loss.data)
            val_loss1.append(loss1.data)
            val_loss2.append(loss2.data)
            val_loss3.append(loss3.data)
            predicted_mask1 = y_preds > best_iou_threshold
            val_acc = iou_pytorch(predicted_mask1.byte(), mask1.squeeze(1).byte())
            val_accuracies.append(val_acc.mean())


        this_epoch_mean_train_loss = torch.mean(torch.stack(train_losses))
        this_epoch_mean_val_loss = torch.mean(torch.stack(val_losses))
        this_epoch_mean_train_acc = torch.mean(torch.stack(train_accuracies))
        this_epoch_mean_val_acc = torch.mean(torch.stack(val_accuracies))
        this_epoch_mean_train_loss1 = torch.mean(torch.stack(train_loss1))
        this_epoch_mean_train_loss2 = torch.mean(torch.stack(train_loss2))
        this_epoch_mean_train_loss3 = torch.mean(torch.stack(train_loss3))
        this_epoch_mean_val_loss1 = torch.mean(torch.stack(val_loss1))
        this_epoch_mean_val_loss2 = torch.mean(torch.stack(val_loss2))
        this_epoch_mean_val_loss3 = torch.mean(torch.stack(val_loss3))

        mean_train_losses.append(this_epoch_mean_train_loss)
        mean_val_losses.append(this_epoch_mean_val_loss)
        mean_train_accuracies.append(this_epoch_mean_train_acc)
        mean_val_accuracies.append(this_epoch_mean_val_acc)
        mean_train_loss1.append(this_epoch_mean_train_loss1)
        mean_train_loss2.append(this_epoch_mean_train_loss2)
        mean_train_loss3.append(this_epoch_mean_train_loss3)
        mean_val_loss1.append(this_epoch_mean_val_loss1)
        mean_val_loss2.append(this_epoch_mean_val_loss2)
        mean_val_loss3.append(this_epoch_mean_val_loss3)

        train_losses_to_csv.extend([this_epoch_mean_train_loss1.item(),this_epoch_mean_train_loss2.item(),this_epoch_mean_train_loss3.item()])
        test_losses_to_csv.extend([this_epoch_mean_val_loss1.item(),this_epoch_mean_val_loss2.item(),this_epoch_mean_val_loss3.item()])

        if overlap == True:
            scheduler.step(this_epoch_mean_val_loss1+this_epoch_mean_val_loss2+this_epoch_mean_val_loss3)    
            early_stopping(this_epoch_mean_val_loss1+this_epoch_mean_val_loss2+this_epoch_mean_val_loss3, model, best_model_fpath)
        else:
            scheduler.step(this_epoch_mean_val_loss1+this_epoch_mean_val_loss2)    
            early_stopping(this_epoch_mean_val_loss1+this_epoch_mean_val_loss2, model, best_model_fpath)

        if early_stopping.early_stop:
            print("Early stopping")
            break


        for param_group in optimizer.param_groups:
            learningRate = param_group['lr']


        # Print Epoch results
        t_end = time.time()
        print('Epoch: {}. Train Loss: {}. Val Loss: {}. Train IoU: {}. Val IoU: {}. Time: {}. LR: {}'
              .format(epoch+1, 
                      this_epoch_mean_train_loss, 
                      this_epoch_mean_val_loss, 
                      this_epoch_mean_train_acc, 
                      this_epoch_mean_val_acc, t_end-t_start, learningRate))
        t_start = time.time()
        print(this_epoch_mean_train_loss1.item(),
              this_epoch_mean_train_loss2.item(),
              this_epoch_mean_train_loss3.item(),
              this_epoch_mean_val_loss1.item(),
              this_epoch_mean_val_loss2.item(),
              this_epoch_mean_val_loss3.item())


    train_losses_to_csv = np.asarray(train_losses_to_csv).reshape((int(len(train_losses_to_csv)/3),3))
    test_losses_to_csv = np.asarray(test_losses_to_csv).reshape((int(len(test_losses_to_csv)/3),3))
    dftrain = pd.DataFrame(train_losses_to_csv) ## convert your array into a dataframe
    dftrain.to_csv('{}_train_losses.csv'.format(best_model_fpath), index=False, header = False)
    dftest = pd.DataFrame(test_losses_to_csv) ## convert your array into a dataframe
    dftest.to_csv('{}_test_losses.csv'.format(best_model_fpath), index=False, header = False)



    totaltime = time.time()-time1
    print("total cost {} hours".format(totaltime/3600))

    mean_train_losses = np.asarray(torch.stack(mean_train_losses).cpu())
    mean_val_losses = np.asarray(torch.stack(mean_val_losses).cpu())
    mean_train_accuracies = np.asarray(torch.stack(mean_train_accuracies).cpu())
    mean_val_accuracies = np.asarray(torch.stack(mean_val_accuracies).cpu())
    fig = plt.figure(figsize=(10,5))
    plt.subplot(1, 2, 1)
    train_loss_series = pd.Series(mean_train_losses)
    val_loss_series = pd.Series(mean_val_losses)
    train_loss_series.plot(label="train_loss")
    val_loss_series.plot(label="validation_loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    train_acc_series = pd.Series(mean_train_accuracies)
    val_acc_series = pd.Series(mean_val_accuracies)
    train_acc_series.plot(label="train_acc")
    val_acc_series.plot(label="validation_acc")
    plt.legend()
    plt.savefig('{}_loss_acc.png'.format(best_model_fpath))

    mean_train_loss1 = np.asarray(torch.stack(mean_train_loss1).cpu())
    mean_train_loss2 = np.asarray(torch.stack(mean_train_loss2).cpu())
    mean_train_loss3 = np.asarray(torch.stack(mean_train_loss3).cpu())
    mean_val_loss1 = np.asarray(torch.stack(mean_val_loss1).cpu())
    mean_val_loss2 = np.asarray(torch.stack(mean_val_loss2).cpu())
    mean_val_loss3 = np.asarray(torch.stack(mean_val_loss3).cpu())
    fig = plt.figure(figsize=(10,5))
    trainloss1 = pd.Series(mean_train_loss1)
    trainloss2 = pd.Series(mean_train_loss2)
    trainloss3 = pd.Series(mean_train_loss3)
    valloss1 = pd.Series(mean_val_loss1)
    valloss2 = pd.Series(mean_val_loss2)
    valloss3 = pd.Series(mean_val_loss3)
    trainloss1.plot(label="trainloss1")
    trainloss2.plot(label="trainloss2")
    trainloss3.plot(label="trainloss3")
    valloss1.plot(label="valloss1")
    valloss2.plot(label="valloss2")
    valloss3.plot(label="valloss3")
    plt.legend()
    # plt.show()
    plt.savefig('{}_losses.png'.format(best_model_fpath))