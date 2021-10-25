import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torch.nn as nn
import timm

from model import FaceModel, Loss, FaceModelFC
from wider_face_dataset import WiderDataset, my_collate
import time

device = torch.device('cuda' if torch.cuda.is_available() else'cpu')
print('you are using ', device)

# hayper parametars
num_epochs = 40
learning_rate = 0.001
batch_size = 48
minLoss = -1

# load data
train_data = WiderDataset(7)
test_data = WiderDataset(7, False)

train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4,
                              collate_fn=my_collate, drop_last=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4,
                             collate_fn=my_collate, drop_last=True)

# create model
model = FaceModelFC()
model.to(device)
model.train()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
print('finished loading model')


# training loop
for epoch in range(num_epochs):
    start_time = time.time()
    model.train()
    epoch_loss = 0
    c = 0
    for x,y in train_dataloader:
        c += 1
        images = x.to(device)
        labels = y.to(device)

        optimizer.zero_grad()
        output = model(images)
        optimizer.zero_grad()
        
        loss = Loss(output, labels)
        loss.backward()
        optimizer.step()

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
            image = x0.to(torch.float32).to(device)
            output = model(image)
            label = y0.to(torch.float32).to(device)
            loss2 += Loss(output, label)

        loss2 /= c_test
        print('loss test : ', loss2.item())
        if loss2 < minLoss or minLoss == -1:
            minLoss = loss2
            torch.save(model.state_dict(), 'models/Best5.pth')
            print('saved!')
        print("time : ", (time.time() - start_time))
        torch.save(model.state_dict(), 'models/Last5.pth')
        with open("models/test5.txt", "a") as file:
            file.write('lr = '+str(learning_rate)+'\n')
            file.write(str(epoch + 1) + '/' + str(num_epochs) + ' loss = ' + str(epoch_loss.item()) + "\n")
            file.write('loss test : ' + str(loss2.item()) + "\n")
            file.write('min loss = ' + str(minLoss) + "\n >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> \n")
        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')






