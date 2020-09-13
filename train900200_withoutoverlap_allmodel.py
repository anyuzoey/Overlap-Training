
# coding: utf-8

# In[1]:

import gc
gc.collect()
import sys
from functions import *
import cmapy
from pytorchtools import EarlyStopping
os.environ["CUDA_VISIBLE_DEVICES"] = "3"  

seeds = [1,236,5193]
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


    # In[3]:


    t_start = time.time()
    seismic = np.load(seismic_path)
    fault = np.load(label_path)
    print("load in {} sec".format(time.time()-t_start))
    print(seismic.shape, fault.shape)
    IL, Z, XL = fault.shape

    modelname = "rcf"
    best_model_fpath = '{}_fullconv_withoutoverlap_seed{}.model'.format(modelname,seed)
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



    t_start = time.time()
    X = []
    Y = []
    for i in range(0,900,1):
        mask = fault[i]
        splits = split_Image(mask, True,top_pad,bottom_pad,left_pad,right_pad,splitsize,stepsize,vertical_splits_number,horizontal_splits_number)
        t = (splits.sum((1,2)) >= pixelThre)
        label_element_index1 = list(compress(range(len(t)), t))
        label_element_index2 = [x+1 for x in label_element_index1]
        Y.extend(splits[label_element_index1])
        Y.extend(splits[label_element_index2]) 

        img = seismic[i]
        splits = split_Image(img, True,top_pad,bottom_pad,left_pad,right_pad,splitsize,stepsize,vertical_splits_number,horizontal_splits_number)
        X.extend(splits[label_element_index1])
        X.extend(splits[label_element_index2])

    print(len(Y))
    print(len(X))
    print(X[0].shape)
    print("read images in {} sec".format(time.time()-t_start))




    t_start = time.time()
    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32)
    print(X.shape)
    print(Y.shape)
    print("read images in {} sec".format(time.time()-t_start))


    # In[22]:


    if len(Y.shape) == 3:
        Y = np.expand_dims(Y, axis=-1)
    if len(X.shape) == 3:
        X = np.expand_dims(X, axis=-1)
    print(X.shape)
    print(Y.shape)

    X_train = X
    Y_train = Y


    t_start = time.time()
    X = []
    Y = []
    for i in range(900,1100,1):
        mask = fault[i]
        splits = split_Image(mask, True,top_pad,bottom_pad,left_pad,right_pad,splitsize,stepsize,vertical_splits_number,horizontal_splits_number)
        t = (splits.sum((1,2)) >= pixelThre)
        label_element_index1 = list(compress(range(len(t)), t))
        label_element_index2 = [x+1 for x in label_element_index1]
        Y.extend(splits[label_element_index1])
        Y.extend(splits[label_element_index2]) 

        img = seismic[i]
        splits = split_Image(img, True,top_pad,bottom_pad,left_pad,right_pad,splitsize,stepsize,vertical_splits_number,horizontal_splits_number)
        X.extend(splits[label_element_index1])
        X.extend(splits[label_element_index2])

    print(len(Y))
    print(len(X))
    print(X[0].shape)
    print("read images in {} sec".format(time.time()-t_start))




    t_start = time.time()
    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32)
    print(X.shape)
    print(Y.shape)
    print("read images in {} sec".format(time.time()-t_start))




    if len(Y.shape) == 3:
        Y = np.expand_dims(Y, axis=-1)
    if len(X.shape) == 3:
        X = np.expand_dims(X, axis=-1)
    print(X.shape)
    print(Y.shape)

    X_val = X
    Y_val = Y

    print("X_train",X_train.shape)
    print("X_val",X_val.shape)

    print("Y_train",Y_train.shape)
    print("Y_val",Y_val.shape)


    X_train = np.moveaxis(X_train,-1,1)
    Y_train = np.moveaxis(Y_train,-1,1)
    X_val = np.moveaxis(X_val,-1,1)
    Y_val = np.moveaxis(Y_val,-1,1)
    print("X_train",X_train.shape)
    print("X_val",X_val.shape)
    print("Y_train",Y_train.shape)
    print("Y_val",Y_val.shape)



    class faultsDataset(torch.utils.data.Dataset):

        def __init__(self,preprocessed_images,train=True, preprocessed_masks=None):
            """
            Args:
                text_file(string): path to text file
                root_dir(string): directory with all train images
            """
            self.train = train
            self.images = preprocessed_images
            self.masks = preprocessed_masks

        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            image = self.images[idx]
            mask = self.masks[idx]
            return (image, mask)




    faults_dataset_train = faultsDataset(X_train, train=True, preprocessed_masks=Y_train)
    faults_dataset_val = faultsDataset(X_val, train=False, preprocessed_masks=Y_val)

    batch_size = 64 

    train_loader = torch.utils.data.DataLoader(dataset=faults_dataset_train, 
                                               batch_size=batch_size, 
                                               shuffle=True)

    val_loader = torch.utils.data.DataLoader(dataset=faults_dataset_val, 
                                               batch_size=batch_size, 
                                               shuffle=True)


    device = torch.device("cuda")
    if modelname =="unet":
        from model_zoo.UNET_fullconv import Unet
        model = Unet()
        print("use model Unet")
    elif modelname == "deeplab":
        from model_zoo.DEEPLAB.deeplab import DeepLab
        model = DeepLab(backbone='mobilenet', num_classes=1, output_stride=16)
        print("use model DeepLab")        
    elif modelname == "rcf":
        from model_zoo.rcf_fullconv import RCF
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
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',factor=0.1, patience=5, verbose=True)


    # In[ ]:


    bceloss = nn.BCELoss()
    mean_train_losses = []
    mean_val_losses = []
    mean_train_accuracies = []
    mean_val_accuracies = []
    t_start = time.time()
    early_stopping = EarlyStopping(patience=patience, verbose=True, delta = 0)
    for epoch in range(epoches):                  
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        labelled_val_accuracies = []

        model.train()
        for images, masks in train_loader: 
            images = Variable(images.cuda())
            masks = Variable(masks.cuda())
            outputs = model(images)

            loss = torch.zeros(1).cuda()
            y_preds = outputs
            if modelname =="unet" or modelname == "deeplab":
                loss = bceloss(outputs, masks)
            elif modelname == "rcf":
                for o in outputs:
                    loss = loss + cross_entropy_loss_RCF(o, masks)
                y_preds = outputs[-1]
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_losses.append(loss.data)
            predicted_mask = y_preds > best_iou_threshold
            train_acc = iou_pytorch(predicted_mask.squeeze(1).byte(), masks.squeeze(1).byte())
            train_accuracies.append(train_acc.mean())        

        model.eval()
        for images, masks in val_loader:
            images = Variable(images.cuda())
            masks = Variable(masks.cuda())
            outputs = model(images)

            loss = torch.zeros(1).cuda()
            y_preds = outputs
            if modelname =="unet" or modelname == "deeplab":
                loss = bceloss(outputs, masks)
            elif modelname == "rcf":
                for o in outputs:
                    loss = loss + cross_entropy_loss_RCF(o, masks)
                y_preds = outputs[-1]
            val_losses.append(loss.data)
            predicted_mask = y_preds > best_iou_threshold
            val_acc = iou_pytorch(predicted_mask.byte(), masks.squeeze(1).byte())
            val_accuracies.append(val_acc.mean())

        mean_train_losses.append(torch.mean(torch.stack(train_losses)))
        mean_val_losses.append(torch.mean(torch.stack(val_losses)))
        mean_train_accuracies.append(torch.mean(torch.stack(train_accuracies)))
        mean_val_accuracies.append(torch.mean(torch.stack(val_accuracies)))

        scheduler.step(torch.mean(torch.stack(val_losses)))    
        early_stopping(torch.mean(torch.stack(val_losses)), model, best_model_fpath)

        if early_stopping.early_stop:
            print("Early stopping")
            break

        for param_group in optimizer.param_groups:
            learningRate = param_group['lr']


        # Print Epoch results
        t_end = time.time()
        print('Epoch: {}. Train Loss: {}. Val Loss: {}. Train IoU: {}. Val IoU: {}. Time: {}. LR: {}'
              .format(epoch+1, torch.mean(torch.stack(train_losses)), torch.mean(torch.stack(val_losses)), torch.mean(torch.stack(train_accuracies)), torch.mean(torch.stack(val_accuracies)), t_end-t_start, learningRate))

        t_start = time.time()



    # In[ ]:
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

    totaltime = time.time()-time1
    print("total cost {} hours".format(totaltime/3600))
