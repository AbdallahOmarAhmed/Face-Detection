import os
import torch
from torch.utils.data import DataLoader
from model import FaceModel, YOLOLoss
from wider_face_dataset import WiderDataset, grid_size, myCollate
import time
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(description='add batch size')
parser.add_argument('batch_size', type=int, help='the batch size of the data loader',)
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else'cpu')
print('you are using', device)

# hayper parametars
num_epochs = 200
learning_rate = 0.001
batch_size = args.batch_size
minLoss = -1
name = 'Root'

train_data = WiderDataset(max_faces=50)
test_data = WiderDataset(train=False, max_faces=50)

train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=myCollate)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=myCollate)

# create model
model = FaceModel('resnet18')
model.to(device)
model.train()
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs, eta_min=5e-6)
myLoss = YOLOLoss(grid_size)
if not os.path.isdir('weights'):
    os.makedirs("weights")
print('finished loading model')
print('_________________________________________________________________\n')

# training loop
for epoch in range(num_epochs):
    start_time = time.time()
    model.train()
    epoch_loss = 0
    c = 0
    for x,y in tqdm(train_dataloader, desc="train batch"):
        c += 1
        images = x.to(device)
        labels = y
        optimizer.zero_grad()
        output = model(images)
        loss = myLoss.forward(output, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss
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
        for x0, y0 in tqdm(test_dataloader, desc="eval batch"):
            c_test += 1
            image = x0.to(device)
            label = y0
            output = model(image)
            loss0 = myLoss.forward(output, label)
            loss2 += loss0
        loss2 /= c_test
        print('loss test : ', loss2.item())
        print("time : ", (time.time() - start_time))
        if loss2 < minLoss or minLoss == -1:
            minLoss = loss2
            torch.save(model.state_dict(), 'weights/min_loss.pth')
            print('saved!')
        torch.save(model.state_dict(), 'weights/Last_epoch.pth')
        with open("weights/logs.txt", "a") as file:
            file.write(str(epoch + 1) + '/' + str(num_epochs) + ' loss = ' + str(epoch_loss.item()) + "\n")
            file.write('loss test : ' + str(loss2.item()) + "\n")
            file.write('min loss = ' + str(minLoss) + "\n >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> \n")
        print('_________________________________________________________________\n')
