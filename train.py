import math

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from model import FaceModel, YOLOLoss
# from wider_face_dataset import WiderDataset, my_collate
from wider_face_dataset_old import WiderDataset, grid_size, myCollate
import time

device = torch.device('cuda' if torch.cuda.is_available() else'cpu')
print(device)

# hayper parametars
num_epochs = 100
learning_rate = 0.001
batch_size = 64
minLoss = -1
name = 'cosine20'
# load data
# train_data = TrainDataset()
# test_data = TestDataset()

train_data = WiderDataset(max_faces=50)
test_data = WiderDataset(train=False, max_faces=50)

train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=myCollate)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=myCollate)

# create model
model = FaceModel('resnet18')
model.to(device)
model.train()
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.8)
myLoss = YOLOLoss(grid_size)

print('finished loading model')
print('__________________________________________________________________________\n')

# training loop
for epoch in range(num_epochs):
    start_time = time.time()
    all_losses = np.array([0., 0., 0.])
    model.train()
    epoch_loss = 0
    c = 0
    print('loading : ', end='', flush=True)
    for x,y in train_dataloader:
        c += 1
        images = x.to(device)
        labels = y
        optimizer.zero_grad()
        output = model(images)
        loss, losses, pred, ans, iou, target = myLoss.forward(output, labels)
        if math.isnan(loss):
            import ipdb;ipdb.set_trace()
            loss = 0
        else:
            loss.backward()
            optimizer.step()
        epoch_loss += loss
        all_losses += losses
        if c % 3 == 0:
            print('|', end='', flush=True)
    print("")
    scheduler.step()
    epoch_loss /= c
    all_losses /= c
    print(epoch + 1, '/', num_epochs, 'loss = ', epoch_loss.item())
    with torch.no_grad():
        print('loss56 :', all_losses[0].item()*16)
        print('loss28 :', all_losses[1].item()*4)
        print('loss14 :', all_losses[2].item())
    model.eval()
    # accuracy
    with torch.no_grad():
        n_correct = 0
        n_total = 0
        loss2 = 0
        c_test = 0
        all_losses2 = np.array([0., 0., 0.])
        for x0, y0 in test_dataloader:
            c_test += 1
            image = x0.to(device)
            label = y0
            output = model(image)
            loss0, losses2, p2, ans2, iou2, target2 = myLoss.forward(output, label)
            if math.isnan(loss0):
                import ipdb;ipdb.set_trace()
                loss0 = 0
            loss2 += loss0
            all_losses2 += losses2
        loss2 /= c_test
        all_losses2 /= c_test
        print('loss test : ', loss2.item())
        with torch.no_grad():
            print('loss56 :', all_losses2[0].item()*16)
            print('loss28 :', all_losses2[1].item()*4)
            print('loss14 :', all_losses2[2].item())
        if c == 20:
            torch.save(model.state_dict(), 'models/Best'+name+'20.pth')
        if loss2 < minLoss or minLoss == -1:
            minLoss = loss2
            torch.save(model.state_dict(), 'models/Best'+name+'.pth')
            print('saved!')
        torch.save(model.state_dict(), 'models/Last'+name+'.pth')
        print("time : ", (time.time() - start_time))
        with open("models/"+name+".txt", "a") as file:
            file.write('lr = '+str(learning_rate)+'\n')
            file.write(str(epoch + 1) + '/' + str(num_epochs) + ' loss = ' + str(epoch_loss.item()) + "\n")
            file.write('loss test : ' + str(loss2.item()) + "\n")
            file.write('min loss = ' + str(minLoss) + "\n >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> \n")
        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')






