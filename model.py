import torch.nn as nn
from torch.nn.modules.module import _addindent
import torch
from torch.utils.data import Dataset
import numpy as np
from data_helper import get_file
from torch.autograd import Variable


INPUT_LENGTH = 1024*200
INPUT_HEIGHT = 257


def torch_summarize(model, show_weights=True, show_parameters=True):
    """Summarizes torch model by showing trainable parameters and weights."""
    tmpstr = model.__class__.__name__ + ' (\n'
    for key, module in model._modules.items():
        # if it contains layers let call it recursively to get params and weights
        if type(module) in [
            torch.nn.modules.container.Container,
            torch.nn.modules.container.Sequential
        ]:
            modstr = torch_summarize(module)
        else:
            modstr = module.__repr__()
        modstr = _addindent(modstr, 2)

        params = sum([np.prod(p.size()) for p in module.parameters()])
        weights = tuple([tuple(p.size()) for p in module.parameters()])

        tmpstr += '  (' + key + '): ' + modstr 
        if show_weights:
            tmpstr += ', weights={}'.format(weights)
        if show_parameters:
            tmpstr +=  ', parameters={}'.format(params)
        tmpstr += '\n'   

    tmpstr = tmpstr + ')'
    return tmpstr


class MalConv(nn.Module):
    def __init__(self,input_height=INPUT_HEIGHT, input_length=INPUT_LENGTH, window_size=500):
        super().__init__()
        embedding_size = 16
        self.embed = nn.Embedding(input_height, embedding_size) 
        self.conv_1 = nn.Conv1d(embedding_size, 128, window_size, stride=window_size, bias=True)

        self.pooling = nn.MaxPool1d(int(input_length/window_size))
        

        self.fc_1 = nn.Linear(128,128)
        self.fc_2 = nn.Linear(128,1)

        self.sigmoid = nn.Sigmoid()        

    def forward(self,x):
        x = self.embed(x)  # Output batch_size, flength, n_embed
        x = torch.transpose(x, 1, 2) # Output batch_size, n_embed, flength
        cnn = self.conv_1(x)
        #gating_weight = self.sigmoid(self.conv_2(x))
        x = self.pooling(cnn)
        x = x.view(-1,128)
        x = self.fc_1(x)
        x = self.fc_2(x)
        x = self.sigmoid(x)
        return x
    
    def predict(self, dataloader):
        val_pred = []
        val_label = []
        for _,val_batch_data in enumerate(dataloader):
            cur_batch_size = val_batch_data[0].size(0)

            exe_input = val_batch_data[0].to(device)
            exe_input = Variable(exe_input.long(),requires_grad=False)

            label = val_batch_data[1].to(device)
            label = Variable(label.float(),requires_grad=False)

            pred = malconv(exe_input)
            val_pred.extend(pred.cpu().data)
            val_label.extend(label)
        return np.array(val_pred), np.array(val_label)
    

class PDFDataSet(Dataset):
    def __init__(self, df, first_n_byte=INPUT_LENGTH):
        self.df = df
        self.first_n_byte = first_n_byte

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        cnt = get_file(row['hash'], row['verdict'])
        tmp = [i+1 for i in cnt[:self.first_n_byte]]
        tmp = tmp+[0]*(self.first_n_byte-len(tmp))
        return np.array(tmp), np.array([row['verdict']])