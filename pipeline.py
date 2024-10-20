import os
import pickle
import numpy as np
import torch
from Backbones.model_factory import get_model
from Backbones.utils import evaluatewp, NodeLevelDataset, evaluate_batch
from training.utils import mkdir_if_missing
from dataset.utils import semi_task_manager
import importlib
import copy
import dgl
import time

def get_pipeline(args):
    if args.minibatch:
        if args.ILmode == 'classIL':
            return pipeline_class_IL_no_inter_edge_minibatch
    else:
        if args.ILmode == 'classIL':
            return pipeline_class_IL_no_inter_edge



def data_prepare(args, dataset):
    torch.cuda.set_device(args.gpu)
    n_cls_so_far = 0
    str_int_tsk = 'inter_tsk_edge' if args.inter_task_edges else 'no_inter_tsk_edge'
    for task, task_cls in enumerate(args.task_seq):
        n_cls_so_far += len(task_cls)
        try:
            if args.load_check:
                subgraph, ids_per_cls, [train_ids, valid_ids, test_ids] = pickle.load(open(f'{args.data_path}/{str_int_tsk}/{args.dataset}_{task_cls}.pkl', 'rb'))
            else:
                if f'{args.dataset}_{task_cls}.pkl' not in os.listdir(f'{args.data_path}/{str_int_tsk}'):
                    subgraph, ids_per_cls, [train_ids, valid_ids, test_ids] = pickle.load(open(f'{args.data_path}/{str_int_tsk}/{args.dataset}_{task_cls}.pkl', 'rb'))
        except:
            print(f'preparing data for task {task}')
            if args.inter_task_edges:
                mkdir_if_missing(f'{args.data_path}/inter_tsk_edge')
                cls_retain = []
                for clss in args.task_seq[0:task + 1]:
                    cls_retain.extend(clss)
                subgraph, ids_per_cls_all, [train_ids, valid_ids, test_ids] = dataset.get_graph(tasks_to_retain=cls_retain)
                with open(f'{args.data_path}/inter_tsk_edge/{args.dataset}_{task_cls}.pkl', 'wb') as f:
                    pickle.dump([subgraph, ids_per_cls_all, [train_ids, valid_ids, test_ids]], f)
            else:
                mkdir_if_missing(f'{args.data_path}/no_inter_tsk_edge')
                subgraph, ids_per_cls, [train_ids, valid_ids, test_ids] = dataset.get_graph(tasks_to_retain=task_cls)
                with open(f'{args.data_path}/no_inter_tsk_edge/{args.dataset}_{task_cls}.pkl', 'wb') as f:
                    pickle.dump([subgraph, ids_per_cls, [train_ids, valid_ids, test_ids]], f)


def pipeline_class_IL_no_inter_edge(args, valid=False):
    epochs = args.epochs if valid else 0
    torch.cuda.set_device(args.gpu)
    dataset = NodeLevelDataset(args.dataset,ratio_valid_test=args.ratio_valid_test,args=args)
    args.d_data, args.n_cls = dataset.d_data, dataset.n_cls
    cls = [list(range(i, i + args.n_cls_per_task)) for i in range(0, args.n_cls-1, args.n_cls_per_task)]
    args.task_seq = cls
    args.n_tasks = len(args.task_seq)
    task_manager = semi_task_manager()
    data_prepare(args, dataset)

    model = get_model(dataset, args).cuda(args.gpu) if valid else None
    life_model = importlib.import_module(f'Baselines.{args.method}_model')
    life_model_ins = life_model.NET(model, task_manager, args) if valid else None
    
    acc_matrix = np.zeros([args.n_tasks, args.n_tasks])
    if args.method == 'tpp':
        prototypes = torch.zeros(args.n_tasks, args.d_data)

    name, ite = args.current_model_save_path
    config_name = name.split('/')[-1]
    subfolder_c = name.split(config_name)[-2]
    save_model_name = f'{config_name}_{ite}'
    save_model_path = f'{args.result_path}/{subfolder_c}val_models/{save_model_name}.pkl'

    if args.method == 'tpp':
        save_proto_name = save_model_name + '_prototypes'
        save_proto_path = f'{args.result_path}/{subfolder_c}val_models/{save_proto_name}.pkl'
    if not valid:
        life_model_ins = pickle.load(open(save_model_path,'rb')).cuda(args.gpu)
        if args.method == 'tpp':
            prototypes = pickle.load(open(save_proto_path,'rb'))
    
    n_cls_so_far = 0
    for task, task_cls in enumerate(args.task_seq):
        n_cls_so_far+=len(task_cls)
        subgraph, ids_per_cls, [train_ids, valid_ids, test_ids] = pickle.load(open(f'{args.data_path}/no_inter_tsk_edge/{args.dataset}_{task_cls}.pkl', 'rb'))
        subgraph = subgraph.to(device='cuda:{}'.format(args.gpu))
        features, labels = subgraph.srcdata['feat'], subgraph.dstdata['label'].squeeze()
        task_manager.add_task(task, n_cls_so_far)
        label_offset1 = task_manager.get_label_offset(task - 1)[1]

        if task == 0 and valid and args.method == 'tpp':
            life_model_ins.pretrain(args, subgraph, features)

        for epoch in range(epochs):
            life_model_ins.observe_il(subgraph, features, labels, task, train_ids, ids_per_cls, label_offset1, dataset)

        if valid and args.method == 'tpp':
            prototypes[task] = life_model_ins.getprototype(subgraph, features, train_ids)
                
        acc_mean = []
        for t in range(task+1):
            subgraph, ids_per_cls, [train_ids, valid_ids_, test_ids_] = pickle.load(open(f'{args.data_path}/no_inter_tsk_edge/{args.dataset}_{args.task_seq[t]}.pkl', 'rb'))
            subgraph = subgraph.to(device='cuda:{}'.format(args.gpu))
            features, labels = subgraph.srcdata['feat'], subgraph.dstdata['label'].squeeze()
            test_ids = valid_ids_ if valid else test_ids_
            ids_per_cls_test = [list(set(ids).intersection(set(test_ids))) for ids in ids_per_cls]

            if args.method == 'tpp':
                if task > 0:
                    taskid = life_model_ins.gettaskid(prototypes, subgraph, features, task+1, test_ids)
                else:
                    taskid = 0
                label_offset1, label_offset2 = task_manager.get_label_offset(int(taskid) - 1)[1], task_manager.get_label_offset(int(taskid))[1]
                labels = labels - label_offset1
            output = life_model_ins.getpred(subgraph, features, task)
            acc = evaluatewp(output, labels, test_ids, cls_balance=args.cls_balance, ids_per_cls=ids_per_cls_test)
            acc_matrix[task][t] = round(acc*100,2)
            acc_mean.append(acc)
            print(f"T{t:02d} {acc*100:.2f}|", end="")

        acc_mean = round(np.mean(acc_mean)*100,2)
        print(f"acc_mean(ID acc): {acc_mean})", end="")
        print() 

    if valid:
        mkdir_if_missing(f'{args.result_path}/{subfolder_c}/val_models')
        with open(save_model_path, 'wb') as f:
            pickle.dump(life_model_ins, f) 
        if args.method == 'tpp':
            with open(save_proto_path, 'wb') as f:
                pickle.dump(prototypes, f)
    print('AP: ', acc_mean)
    backward = []
    for t in range(args.n_tasks-1):
        b = acc_matrix[args.n_tasks-1][t]-acc_matrix[t][t]
        backward.append(round(b, 2))
    mean_backward = round(np.mean(backward),2)
    print('AF: ', mean_backward)
    print('\n')
    return acc_mean, mean_backward, acc_matrix


def pipeline_class_IL_no_inter_edge_minibatch(args, valid=False):
    epochs = args.epochs if valid else 0
    torch.cuda.set_device(args.gpu)
    dataset = NodeLevelDataset(args.dataset,ratio_valid_test=args.ratio_valid_test,args=args)
    args.d_data, args.n_cls = dataset.d_data, dataset.n_cls
    cls = [list(range(i, i + args.n_cls_per_task)) for i in range(0, args.n_cls-1, args.n_cls_per_task)]
    args.task_seq = cls
    args.n_tasks = len(args.task_seq)
    task_manager = semi_task_manager()
    data_prepare(args, dataset)

    model = get_model(dataset, args).cuda(args.gpu) if valid else None
    life_model = importlib.import_module(f'Baselines.{args.method}_model')
    life_model_ins = life_model.NET(model, task_manager, args) if valid else None

    acc_matrix = np.zeros([args.n_tasks, args.n_tasks])

    if args.method == 'tpp':
        prototypes = torch.zeros(args.n_tasks, args.d_data)

    name, ite = args.current_model_save_path
    config_name = name.split('/')[-1]
    subfolder_c = name.split(config_name)[-2]
    save_model_name = f'{config_name}_{ite}'
    save_model_path = f'{args.result_path}/{subfolder_c}val_models/{save_model_name}.pkl'

    if args.method == 'tpp':
        save_proto_name = save_model_name + '_prototypes'
        save_proto_path = f'{args.result_path}/{subfolder_c}val_models/{save_proto_name}.pkl'
    if not valid:
        life_model_ins = pickle.load(open(save_model_path,'rb')).cuda(args.gpu)
        if args.method == 'tpp':
            prototypes = pickle.load(open(save_proto_path,'rb'))

    n_cls_so_far = 0
    for task, task_cls in enumerate(args.task_seq):
        n_cls_so_far += len(task_cls)
        subgraph, ids_per_cls, [train_ids, valid_idx, test_ids] = pickle.load(open(f'{args.data_path}/no_inter_tsk_edge/{args.dataset}_{task_cls}.pkl', 'rb'))
        subgraph = subgraph.to(device='cuda:{}'.format(args.gpu))
        features, labels = subgraph.srcdata['feat'], subgraph.dstdata['label'].squeeze()
        task_manager.add_task(task, n_cls_so_far)
        label_offset1 = task_manager.get_label_offset(task - 1)[1]

        if task == 0 and valid and args.method == 'tpp':
            life_model_ins.pretrain(args, subgraph, features, batch_size = args.batch_size)

        for epoch in range(epochs):
            life_model_ins.observe_il(subgraph, features, labels, task, train_ids, ids_per_cls, label_offset1)
            torch.cuda.empty_cache() 

        if valid and args.method == 'tpp':
            prototypes[task] = life_model_ins.getprototype(subgraph, features, train_ids)
 
        acc_mean = []
        for t in range(task + 1):
            subgraph, ids_per_cls, [train_ids, valid_ids_, test_ids_] = pickle.load(open(f'{args.data_path}/no_inter_tsk_edge/{args.dataset}_{args.task_seq[t]}.pkl', 'rb'))
            subgraph = subgraph.to(device='cuda:{}'.format(args.gpu))
            test_ids = valid_ids_ if valid else test_ids_
            ids_per_cls_test = [list(set(ids).intersection(set(test_ids))) for ids in ids_per_cls]
            features, labels = subgraph.srcdata['feat'], subgraph.dstdata['label'].squeeze()
            if args.method == 'tpp':
                if task == 0:
                    taskid = 0
                else:
                    taskid = life_model_ins.gettaskid(prototypes, subgraph, features, task+1, test_ids)
                label_offset1 = task_manager.get_label_offset(int(taskid) - 1)[1]
                labels = labels - label_offset1
            output = life_model_ins.getpred(subgraph, features, taskid)    
            acc = evaluatewp(output, labels, test_ids, cls_balance=args.cls_balance, ids_per_cls=ids_per_cls_test)
            acc_matrix[task][t] = round(acc * 100, 2)
            acc_mean.append(acc)
            print(f"T{t:02d} {acc * 100:.2f}|", end="")

        acc_mean = round(np.mean(acc_mean) * 100, 2)
        print(f"acc_mean(ID acc): {acc_mean})", end="")        
        print()
        
    if valid:
        mkdir_if_missing(f'{args.result_path}/{subfolder_c}/val_models')
        with open(save_model_path, 'wb') as f:
            pickle.dump(life_model_ins, f)
        if args.method == 'tpp':
            with open(save_proto_path, 'wb') as f:
                pickle.dump(prototypes, f)

    print('AP: ', acc_mean)
    backward = []
    for t in range(args.n_tasks - 1):
        b = acc_matrix[args.n_tasks - 1][t] - acc_matrix[t][t]
        backward.append(round(b, 2))
    mean_backward = round(np.mean(backward), 2)
    print('AF: ', mean_backward)
    print('\n')
    return acc_mean, mean_backward, acc_matrix
