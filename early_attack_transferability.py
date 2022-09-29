
import torch
import time
from torch import optim
import os
import random
import numpy as np
import torch.nn
from torch.autograd import Variable
import csv
import aux_funcs as af
import network_architectures as arcs
import time
from architectures.CNNs.VGG import VGG
from torch.utils.data import TensorDataset, DataLoader
import torch.utils.data as torchdata

dataset = af.get_dataset('cifar10',10000)
data_arr=next(iter(dataset.test_loader))[0].numpy()
#print(data_arr.shape)
#data_arr=data_arr[0:5000].tolist()

test_data=next(iter(dataset.test_loader))[0].numpy()[5000:]
test_labels=next(iter(dataset.test_loader))[1].numpy()[5000:]

#print()
tensor_x = torch.Tensor(test_data)  # transform to torch tensor
tensor_y = torch.Tensor(test_labels)

my_dataset = TensorDataset(tensor_x, tensor_y)  # create your datset
# my_dataloader = DataLoader(my_dataset)
device = af.get_pytorch_device(gpu_id=2)
test_loader = torchdata.DataLoader(my_dataset, batch_size=1, shuffle=False, num_workers=4)



datasets=['cifar10','cifar100']
sdn_names=['vgg16bn','mobilenet','resnet56']
#datasets=['cifar10']
#sdn_names=['mobilenet']
#alphas=[1,20,40]
alphas=[0.1,0.01,0.001]
for dataset in datasets:
    for name in sdn_names:

        orig = np.load('attack/' + dataset + '_' + name + '_' + str(1) + '_orig.npy')

        adv = np.load('attack/' + dataset + '_' + name + '_' + str(1) + '_adv.npy')

        test_labels = [0 for i in range(len(orig))]

        # print()
        tensor_x = torch.Tensor(orig)  # transform to torch tensor
        tensor_y = torch.Tensor(adv)


        for name2 in sdn_names:
            sdn_name = dataset + '_' + name2 + '_' + 'sdn_sdn_training'


            sdn_model, sdn_params = arcs.load_model(models_path, sdn_name, epoch=-1)
            sdn_model.to(device)
            sdn_model.eval()



            #my_dataset = TensorDataset(tensor_x, tensor_y)  # create your datset
            # my_dataloader = DataLoader(my_dataset)
            #device = af.get_pytorch_device(gpu_id=2)
            #test_loader = torchdata.DataLoader(my_dataset, batch_size=1, shuffle=False, num_workers=4)

            idx2 = 0
            f_cnt = 0
            #orig = []
            #adv = []

            for i in range(len(orig)):
                op1=sdn_model(torch.Tensor(orig[i]).to(device))
                out_label=torch.argmax(op1[len(op1)-1])

                op2 = sdn_model(torch.Tensor(adv[i]).to(device))

                lst = []
                cnt=0
                for idx in range(len(op2)):
                    tmp = torch.argmax(op2[idx])

                    if (tmp.item() == out_label):
                        if (idx == len(op2) - 1):
                            cnt += 1
                            lst.append(True)
                        else:
                            lst.append(False)
                    else:
                        if (idx != len(op2) - 1):
                            cnt += 1
                            lst.append(True)
                        else:
                            lst.append(False)
                    if (cnt == len(op2)):
                        f_cnt += 1

                filename = 'trasferability/'+dataset+"_"+name+"_"+name2+"_"+".csv"
                #fields = [dataset, name, name2, str(f_cnt), str(idx2)]
                with open(filename, 'a') as csvfile:
                    # creating a csv writer object
                    csvwriter = csv.writer(csvfile)

                    # writing the fields
                    csvwriter.writerow(lst)

            #print(dataset, " ", name)
            #print(lst)
                idx2+=1
            filename='transferability_cnt.csv'
            fields = [dataset, name, name2, str(f_cnt), str(idx2)]
            with open(filename, 'a') as csvfile:
                # creating a csv writer object
                csvwriter = csv.writer(csvfile)

                # writing the fields
                csvwriter.writerow(fields)

                #time.sleep(1)