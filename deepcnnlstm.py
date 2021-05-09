'''
Reference:

https://www.medrxiv.org/content/10.1101/2020.06.18.20134718v1.full.pdf
https://www.sciencedirect.com/science/article/pii/S2352914820305621
'''

import torch.nn as nn
import torch.nn.functional as F

class DeepCNNLSTM(nn.Module):
    def __init__(self, n_classes=2):
    	super(DeepCNNLSTM, self).__init__()
    	self.conv_3x64 = nn.Conv2d(3, 64, kernel_size=1)
    	self.conv_64x64 = nn.Conv2d(64, 64, kernel_size=1)
    	self.conv_64x128 = nn.Conv2d(64, 128, kernel_size=1)
    	self.conv_128x128 = nn.Conv2d(128, 128, kernel_size=1)
    	self.conv_128x256 = nn.Conv2d(128, 256, kernel_size=1)
    	self.conv_256x256 = nn.Conv2d(256, 256, kernel_size=1)
    	self.conv_256x512 = nn.Conv2d(256, 512, kernel_size=1)
    	self.conv_512x512 = nn.Conv2d(512, 512, kernel_size=1)
    	self.pool = nn.MaxPool2d(2, stride=2)
    	self.dropout = nn.Dropout2d(p=0.25)
    	self.lstm = nn.LSTM(49, 49, batch_first=True)
    	self.fc = nn.Linear(49 * 512, 64)
    	self.classifier = nn.Linear(64, n_classes)

    def forward(self, x):
    	out = F.relu(self.conv_3x64(x))
    	out = F.relu(self.conv_64x64(out))
    	out = self.dropout(self.pool(out))
    	out = F.relu(self.conv_64x128(out))
    	out = F.relu(self.conv_128x128(out))
    	out = self.dropout(self.pool(out))
    	out = F.relu(self.conv_128x256(out))
    	out = F.relu(self.conv_256x256(out))
    	out = self.dropout(self.pool(out))
    	out = F.relu(self.conv_256x512(out))
    	out = F.relu(self.conv_512x512(out))
    	out = F.relu(self.conv_512x512(out))
    	out = self.dropout(self.pool(out))
    	out = F.relu(self.conv_512x512(out))
    	out = F.relu(self.conv_512x512(out))
    	out = F.relu(self.conv_512x512(out))
    	out = self.dropout(self.pool(out))
    	out = out.view(out.size(0), out.size(1), -1)
    	out, _ = self.lstm(out)
    	out = out.reshape(out.size(0), -1)
    	out = self.fc(out)
    	out = self.classifier(out)
    	out = F.softmax(out, dim=1)
    	return out

