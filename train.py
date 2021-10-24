import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torch.nn as nn
import timm

from model import FaceModel, Loss
from wider_face_dataset import WiderDataset, my_collate
import time

device = torch.device('cuda' if torch.cuda.is_available() else'cpu')
print('you are using ', device)

# hayper parametars
num_epochs = 40
learning_rate = 0.0001
batch_size = 50
minLoss = -1

# load data
train_data = WiderDataset(7)
test_data = WiderDataset(7, False)

train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4,
                              collate_fn=my_collate, drop_last=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4,
                             collate_fn=my_collate, drop_last=True)

# create model
model = FaceModel('resnet18')
optimaizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
model.to(device)
print('finished loading model')


# training loop
for epoch in range(num_epochs):
    start_time = time.time()
    model.train()
    epoch_loss = 0
    c = 0
    for x,y in train_dataloader:
        c += 1
        x = x.permute(0,3,1,2)
        images = x.to(torch.float32).to(device)
        labels = y.to(torch.float32).to(device)
        output = model(images)
        loss = Loss(output, labels)

        # backward
        loss.backward()
        optimaizer.zero_grad()
        optimaizer.step()
        with torch.no_grad():
            epoch_loss += loss

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
            x0 = x0.permute(0, 3, 1, 2)
            image = x0.to(device)
            output = model(image)
            label = y0.to(device)
            loss2 += Loss(output, label)

        loss2 /= c_test
        print('loss test : ', loss2.item())
        if loss2 < minLoss or minLoss == -1:
            minLoss = loss2
            torch.save(model.state_dict(), 'models/Best2.pth')
            print('saved!')
        print("time : ", (time.time() - start_time))
        torch.save(model.state_dict(), 'models/Last2.pth')
        with open("models/test2.txt", "a") as file:
            file.write('lr = '+str(learning_rate)+'\n')
            file.write(str(epoch + 1) + '/' + str(num_epochs) + ' loss = ' + str(epoch_loss.item()) + "\n")
            file.write('loss test : ' + str(loss2.item()) + "\n")
            file.write('min loss = ' + str(minLoss) + "\n >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> \n")
        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')






