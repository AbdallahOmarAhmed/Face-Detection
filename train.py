import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torch.nn as nn
import timm

from model import FaceModel, Loss, FaceModelFC, LossVec
# from wider_face_dataset import WiderDataset, my_collate
from wider_face_dataset_old import WiderDataset
import time

device = torch.device('cuda' if torch.cuda.is_available() else'cpu')
print(device)

# hayper parametars
num_epochs = 80
learning_rate = 0.0003
batch_size = 48
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
model = FaceModelFC()
model.to(device)
model.train()
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)

print('finished loading model')


# training loop
for epoch in range(num_epochs):
    start_time = time.time()
    model.train()
    epoch_loss = 0
    c = 0
    #print('loading : ', end='',flush=True)
    for x,y in train_dataloader:
        c += 1
        x = x.permute(0, 3, 1, 2)
        images = x.to(torch.float32).to(device)
        labels = y.to(torch.float32).to(device)

        output = model(images)

        optimizer.zero_grad()
        loss = LossVec(output, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss/batch_size
        #print('#', end='', flush=True)
    #print("")
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
        for x, y in test_dataloader:
            c_test += 1
            x = x.permute(0, 3, 1, 2)
            y = y.to(device)
            image = x.to(device)

            output = model(image)
            loss2 += LossVec(output, y)/batch_size

        loss2 /= c_test
        print('loss test : ', loss2.item())
        if loss2 < minLoss or minLoss == -1:
            minLoss = loss2
            torch.save(model.state_dict(), 'models/test1.pth')
            print('saved!')
        print("time : ", (time.time() - start_time))
        torch.save(model.state_dict(), 'models/test2.pth')
        with open("models/test0.txt", "a") as file:
            file.write('lr = '+str(learning_rate)+'\n')
            file.write(str(epoch + 1) + '/' + str(num_epochs) + ' loss = ' + str(epoch_loss.item()) + "\n")
            file.write('loss test : ' + str(loss2.item()) + "\n")
            file.write('min loss = ' + str(minLoss) + "\n >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> \n")
        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')






