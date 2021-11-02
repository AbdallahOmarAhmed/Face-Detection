import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torch.nn as nn
import timm
from loss import YOLOLoss

from model import FaceModel, Loss, FaceModelFC, LossVec
# from wider_face_dataset import WiderDataset, my_collate
from wider_face_dataset_old import WiderDataset
import time

device = torch.device('cuda' if torch.cuda.is_available() else'cpu')
print(device)

# hayper parametars
num_epochs = 80
learning_rate = 0.0003
batch_size = 50
minLoss = -1
# load data
# train_data = TrainDataset()
# test_data = TestDataset()
#
train_data = WiderDataset(5)
test_data = WiderDataset(5, False)

train_dataloader = DataLoader(train_data, batch_size=batch_size,shuffle=True, num_workers=4)
test_dataloader = DataLoader(test_data, batch_size=batch_size,shuffle=False, num_workers=4)

# create model
model = FaceModel('resnet18')
model.to(device)
model.train()
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)
myLoss = YOLOLoss(7)

print('finished loading model')
print('___________________________________________________________________________\n')

# training loop
for epoch in range(num_epochs):
    start_time = time.time()
    model.train()
    epoch_loss = 0
    c = 0
    print('loading : ', end='',flush=True)
    for x,y in train_dataloader:
        c += 1
        images = x.to(device)
        labels = y.to(device)

        optimizer.zero_grad()
        output = model(images)
        loss = myLoss.forward(output, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss
        print('|', end='', flush=True)
    print("")
    scheduler.step()
    epoch_loss /= c
    print(epoch + 1, '/', num_epochs, 'loss = ', epoch_loss.item())
    model.eval()
    # accuracy
    with torch.no_grad():
        n_correct = 0
        n_total = 0
        loss2 = 0
        c_test = 0
        for x0, y0 in test_dataloader:
            c_test += 1
            image = x0.to(device)
            label = y0.to(device)

            output = model(image)
            loss2 += myLoss.forward(output, label)

        loss2 /= c_test
        print('loss test : ', loss2.item())
        if loss2 < minLoss or minLoss == -1:
            minLoss = loss2
            torch.save(model.state_dict(), 'models/BestNoFc2.pth')
            print('saved!')
        torch.save(model.state_dict(), 'models/LastNoFc2.pth')
        print("time : ", (time.time() - start_time))
        with open("models/NoFc2.txt", "a") as file:
            file.write('lr = '+str(learning_rate)+'\n')
            file.write(str(epoch + 1) + '/' + str(num_epochs) + ' loss = ' + str(epoch_loss.item()) + "\n")
            file.write('loss test : ' + str(loss2.item()) + "\n")
            file.write('min loss = ' + str(minLoss) + "\n >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> \n")
        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')






