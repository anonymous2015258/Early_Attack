#d

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

test_loader = torchdata.DataLoader(my_dataset, batch_size=1, shuffle=False, num_workers=4)


def normalize(t):
    n = t.clone()
    for i in range(3):
        n[:, i, :, :] = (t[:, i, :, :] - mean[i]) / std[i]
    return n
def denormalize(t):
    n = t.clone()
    for i in range(3):
        n[:, i, :, :] = t[:, i, :, :] * std[i] + mean[i]
    return n
device = af.get_pytorch_device(gpu_id=2)


def tanh_rescale(x, x_min=-1.7, x_max=2.05):
    return (torch.tanh(x) + 1) * 0.5 * (x_max - x_min) + x_min


def reduce_sum(x, keepdim=True):
    # silly PyTorch, when will you get proper reducing sums/means?
    for a in reversed(range(1, x.dim())):
        x = x.sum(a, keepdim=keepdim)
    return x


def l2_dist(x, y, keepdim=True):
    d = (x - y) ** 2
    return reduce_sum(d, keepdim=keepdim)


def loss_op(loss1, dist, scale_const):
    #loss1 = (- 1) * output
    loss1 = torch.sum(scale_const * loss1)
    loss2 = dist.sum()
    loss = loss1 + loss2
    return loss

def cw_attack(model, images, labels, alpha,c=50, iters=1000):
    bc = 1
    w = torch.rand([bc, 3, 32, 32], requires_grad=True,device=device)
    input_adv = torch.rand([bc, 3, 32, 32], requires_grad=True,device=device)
    optimizer = optim.Adam([w], lr=0.01)
    c = torch.tensor(c, requires_grad=False,device=device)

    outputs = model(images)

    out_label = torch.argmax(outputs[len(outputs) - 1]).item()
    m = torch.nn.ReLU()
    m2 = torch.nn.Softmax()
    for i in range(iters):
        input_adv = normalize((torch.tanh(w) + 1) / 2)
        outputs = model(input_adv)

        tmp = torch.tensor(0.0).to(device)
        final_layer_sm = m2(outputs[len(outputs) - 1][0])
        #print(final_layer_sm[out_label])
        # print(outputs[len(outputs) - 1].size().item())
        for j in range(10):
            # print(out_label,' ',j)
            # print(j == out_label)
            if j == out_label:
                continue
            else:
                # print(outputs[len(outputs) - 1])
                val = final_layer_sm[out_label] - final_layer_sm[j]
                #val = final_layer_sm[j]
                tmp = torch.add(tmp, m(val))
        # print('in')
        tmp2 = torch.tensor(0.0).to(device)
        for k in range(len(outputs) - 1):
            # print(outputs[k][0])
            layer_sm = m2(outputs[k][0])
            # print(layer_sm[out_label])
            tmp2 = torch.add(layer_sm[out_label], tmp2)
            #print(tmp2)
        # out_label=outputs[len(outputs) - 1]
        #time.sleep(2)
        #alpha=1
        cost = (-alpha*tmp + tmp2).to(device)

        loss1 = l2_dist(images, input_adv)
        loss = loss_op(cost, loss1, c)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        op2 = model(input_adv)
        cnt=0
        lst=[]
        for idx in range(len(op2)):
            tmp = torch.argmax(op2[idx])
            '''if(idx==5 or idx==5):
                print(torch.max(m2(op2[idx])))'''
            if (tmp.item() == out_label):
                if(idx==len(op2)-1):
                    cnt+=1
                    lst.append(True)
                else:
                    lst.append(False)
            else:
                if (idx != len(op2) - 1):
                    cnt += 1
                    lst.append(True)
                else:
                    lst.append(False)
        #print(cnt)
        #print(len(op2))
        #print(lst)
        #print("-----")
        if(cnt==len(op2) and i>=200):
            #print(l2_dist(images, input_adv))
            return input_adv,out_label
    #print(l2_dist(images, input_adv))
    return input_adv,out_label

'''def pgd_attack(model, images, labels, eps=0.3, alpha=2 / 255, iters=40):
    images = images.to(device)
    #labels = labels.to(device)
    labels = labels.type(torch.LongTensor).to(device)
    loss = torch.nn.CrossEntropyLoss()

    ori_images = images.data

    outputs = model(normalize(images))


    out_label = torch.argmax(outputs[len(outputs) - 1]).item()
    print('out_label', out_label)

    m=torch.nn.ReLU()
    m2=torch.nn.Softmax()
    for i in range(iters):
        #images.requires_grad = True

        images=Variable(images, requires_grad=True).to(device)
        #outputs = model(normalize(images))
        model.zero_grad()
        outputs = model(normalize(images))
        #print(out_label)
        sm=m2(outputs[len(outputs) - 1][0])
        
        # print(outputs[len(outputs) - 1])
        # print(labels)
        tmp=torch.tensor(0.0).to(device)
        final_layer_sm=m2(outputs[len(outputs) - 1][0])

        #print(outputs[len(outputs) - 1].size().item())
        for j in range(10):
            #print(out_label,' ',j)
            #print(j == out_label)
            if j == out_label:
                continue
            else:
                #print(outputs[len(outputs) - 1])
                val=final_layer_sm[out_label]-final_layer_sm[j]
                tmp=torch.add(tmp,m(val))
        #print('in')
        tmp2 = torch.tensor(0.0).to(device)
        for k in range(len(outputs)-1):
            #print(outputs[k][0])
            layer_sm=m2(outputs[k][0])
            #print(layer_sm[out_label])
            tmp2=torch.add(layer_sm[out_label],tmp2)
            print(tmp2)
        #out_label=outputs[len(outputs) - 1]
        time.sleep(2)
        cost = (tmp+10*tmp2).to(device)

        #model.zero_grad()

        cost.backward()

        adv_images = images + alpha * images.grad.sign()
        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        images = torch.clamp(ori_images + eta, min=0, max=1).detach_()

    return images, out_label'''





models_path='/glusterfs/data/mxh170530/Shallow-Deep-Networks-master/networks/1221/'

datasets=['cifar10','cifar100']
sdn_names=['vgg16bn','mobilenet','resnet56']
#datasets=['cifar10']
#sdn_names=['vgg16bn','mobilenet']
#alphas=[1,20,40]
alphas=[0.1,0.01,0.001]
for dataset in datasets:

    if(dataset=='cifar10'):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        mean = [0.507, 0.487, 0.441]
        std = [0.267, 0.256, 0.276]
    dataset2 = af.get_dataset(dataset, 10000)

    test_data = next(iter(dataset2.test_loader))[0].numpy()[5000:]
    test_labels = next(iter(dataset2.test_loader))[1].numpy()[5000:]

    # print()
    tensor_x = torch.Tensor(test_data)  # transform to torch tensor
    tensor_y = torch.Tensor(test_labels)

    my_dataset = TensorDataset(tensor_x, tensor_y)  # create your datset
    # my_dataloader = DataLoader(my_dataset)

    test_loader = torchdata.DataLoader(my_dataset, batch_size=1, shuffle=False, num_workers=4)
    for name in sdn_names:
        for alpha in alphas:
            if((dataset=='cifar10' and name=='vgg16bn') or (dataset=='cifar10' and name=='mobilenet')):
                continue

            sdn_name = dataset + '_' + name + '_' + 'sdn_sdn_training'

            sdn_model, sdn_params = arcs.load_model(models_path, sdn_name, epoch=-1)
            sdn_model.to(device)
            sdn_model.eval()

            idx2 = 0
            f_cnt = 0
            orig=[]
            adv=[]
            for batch in test_loader:
                if (idx2 == 100):
                    break
                # sdn_model.forward = sdn_model.early_exit
                # sdn_model2.confidence_threshold = 0.9
                b_x = batch[0].to(device)
                b_y = batch[1].to(device)
                input_var: object = Variable(b_x, requires_grad=True).to(device)
                labels = Variable(b_y, requires_grad=False).to(device)
                # output = sdn_model(b_x)

                # adv_ip,out_label = pgd_attack(sdn_model, denormalize(input_var), labels)

                # op2 = sdn_model(normalize(adv_ip))

                adv_ip, out_label = cw_attack(sdn_model, input_var, labels,alpha)
                orig.append(input_var.tolist())
                adv.append(adv_ip.tolist())
                op2 = sdn_model(adv_ip)
                lst = []
                '''for o in op2:
                    tmp=torch.argmax(o)
                    if(tmp.item()==out_label):
                        lst.append(True)
                    else:
                        lst.append(False)
                print(lst)
                time.sleep(1)'''

                cnt = 0
                #print(op2)
                for idx in range(len(op2)):
                    tmp = torch.argmax(op2[idx])
                    if (tmp.item() == out_label):
                        if (idx == len(op2) - 1):
                            cnt += 1
                        # lst.append(True)
                    else:
                        if (idx != len(op2) - 1):
                            cnt += 1
                if (cnt == len(op2)):
                    f_cnt += 1
                idx2 += 1


                print(alpha,' f_cnt_',f_cnt,' ','idx2_',idx2)

            filename = "early_attack_results2.csv"

            # writing to csv file

            np.save('attack/'+dataset+'_'+name+'_'+str(alpha)+'_orig.npy',np.asarray(orig))

            np.save('attack/' + dataset + '_' + name + '_' + str(alpha) + '_adv.npy', np.asarray(adv))

            fields = [dataset, name, str(alpha),str(f_cnt),  str(idx)]
            with open(filename, 'a') as csvfile:
                # creating a csv writer object
                csvwriter = csv.writer(csvfile)

                # writing the fields
                csvwriter.writerow(fields)

