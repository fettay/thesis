from data_helper import files_df, split_train_test
from torch.utils.data import DataLoader
from torch.nn.modules.module import _addindent
import torch
import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import time
import logging
from uuid import uuid4
from model import INPUT_LENGTH, INPUT_HEIGHT, torch_summarize, MalConv, PDFDataSet
import argparse


TEST_VALID_RATIO = 0.1
VALID_RATIO = 0.2
BATCH_SIZE = 64
NB_EPOCHS = 4

learning_rate = 1e-3
display_step = 15
test_step = 150
training_id = str(uuid4())

checkpoint_file = 'Chkp/%s' % training_id

logfile = 'logs/%s.out' % training_id
logger = logging.getLogger()
logger.warning('Writing logs in : %s' % logfile)

logging.basicConfig(level=logging.DEBUG, filename=logfile)
logger.debug('Training id: %s' % training_id)

    
def run(gpu=None):
    # Preprocessing
    if gpu is None:
        gpu = "cuda:0"
    device = torch.device(gpu if torch.cuda.is_available() else "cpu")
    logger.debug('Used device %s' % device)
    df = files_df().sample(frac=1, random_state=42)
    df_train, df_test = split_train_test(df, TEST_VALID_RATIO)
    df_test, df_valid = split_train_test(df_test, VALID_RATIO)
    logger.debug('Train size %d, Valid size %d,  Test size %d' % (df_train.shape[0],
                                                               df_valid.shape[0], df_test.shape[0]))
    
    dataloader = DataLoader(PDFDataSet(df_train), batch_size=BATCH_SIZE, shuffle=False)
    validloader = DataLoader(PDFDataSet(df_valid), batch_size=BATCH_SIZE, shuffle=False)

    malconv = MalConv()
    #bce_loss = nn.BCEWithLogitsLoss()
    bce_loss = nn.BCELoss()
    adam_optim = optim.Adam([{'params':malconv.parameters()}],lr=learning_rate)
    sigmoid = nn.Sigmoid()
    
    logger.debug(torch_summarize(malconv))

    malconv = malconv.to(device)
    bce_loss = bce_loss.to(device)
    sigmoid = sigmoid.to(device)


    step_msg = 'Step:{} | Loss:{:.6f} | Acc:{:.4f} | Time:{:.2f}'
    valid_msg = 'Val_loss:{:.6f} | Val_acc:{:.4f}'
    history = {}
    history['tr_loss'] = []
    history['tr_acc'] = []

    logger.debug('step,tr_loss, tr_acc, val_loss, val_acc, time\n')

    valid_best_acc = 0.0
    total_step = 0
    step_cost_time = 0
    max_step = (int(df_train.shape[0] / BATCH_SIZE) + 1) * NB_EPOCHS   # Last number is the number of epochs
        
    while total_step < max_step:

        # Training 
        for step,batch_data in enumerate(dataloader):
            start = time.time()

            adam_optim.zero_grad()

            cur_batch_size = batch_data[0].size(0)

            exe_input = batch_data[0].to(device)
            exe_input = Variable(exe_input.long(),requires_grad=False)
            label = batch_data[1].to(device)
            label = Variable(label.float(),requires_grad=False)
            pred = malconv(exe_input)
            loss = bce_loss(pred,label)
            loss.backward()
            adam_optim.step()
            history['tr_loss'].append(loss.cpu().data.item())
            history['tr_acc'].extend(list(label.cpu().data.numpy().astype(int)==(pred.cpu().data.numpy()+0.5).astype(int)))

            step_cost_time = time.time()-start

            if step%display_step == 0:
                logger.debug(step_msg.format(total_step,np.mean(history['tr_loss']),
                                      np.mean(history['tr_acc']),step_cost_time))
                history['tr_loss'] = []
                history['tr_acc'] = []

            # Interupt for validation
            if total_step%test_step ==0:


                # Testing
                history['val_loss'] = []
                history['val_acc'] = []
                history['val_pred'] = []

                for _,val_batch_data in enumerate(validloader):
                    cur_batch_size = val_batch_data[0].size(0)

                    exe_input = val_batch_data[0].to(device)
                    exe_input = Variable(exe_input.long(),requires_grad=False)

                    label = val_batch_data[1].to(device)
                    label = Variable(label.float(),requires_grad=False)

                    pred = malconv(exe_input)
                    loss = bce_loss(pred,label)

                    history['val_loss'].append(loss.cpu().data.item())
                    history['val_acc'].extend(list(label.cpu().data.numpy().astype(int)==(pred.cpu().data.numpy()+0.5).astype(int)))

                logger.info(valid_msg.format(np.mean(history['val_loss']),np.mean(history['val_acc'])))
                if valid_best_acc < np.mean(history['val_acc']):
                    valid_best_acc = np.mean(history['val_acc'])
                    torch.save(malconv, checkpoint_file)
                    logger.debug('Checkpoint saved at %s' % checkpoint_file)

            total_step += 1
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default=None)
    args = parser.parse_args()
    run(args.gpu)
