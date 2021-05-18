# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu

import os

ROOT_DATASET =  'G://YMY//DATASET//'
#'G://YMY//DATASET//UCF101//ucf101_jpegs_256//' 'G://YMY//DATASET//UCF101//ucf101_tvl1_flow//'
# 'G://YMY//DATASET//HMDB51//hmdb51_tvl1_flow//' 'G://YMY//DATASET//HMDB51//hmdb51_jpegs_256//' 

def return_ucfcrime(modality):
    filename_categories = '/media/wzy/WZYPassport/UCF_Crime_other/labels/ClassIDs.txt' #D:/Users/wzy\PycharmProjects\code\CVPR2018_Pytorch_0605_copy\data/ucfcrime_splits\ClassIDs.txt#/media/wzy/WZYPassport/UCF_Crime_other/labels/ClassIDs.txt
    if modality == 'RGB':
        root_data = ROOT_DATASET + 'UCF_Crime_other\jpg_cv'
        filename_imglist_train =  ROOT_DATASET + '/UCF_Crime_other/file_list_cv1/ucfcrime_rgb_train_split_5.txt'#D:/Users\wzy\PycharmProjects\code/UCF_Crime_other\file_list_cv/ucfcrime_rgb_train_split_1.txt
        filename_imglist_val = ROOT_DATASET + '/UCF_Crime_other/file_list_cv1/ucfcrime_rgb_val_split_5.txt'#D:/Users\wzy\PycharmProjects\code/UCF_Crime_other\file_list_cv/ucfcrime_rgb_val_split_1.txt
        prefix = 'img_{:05d}.jpg'
    elif modality == 'Flow':
        root_data = ROOT_DATASET + 'UCF101/jpg'
        filename_imglist_train = ROOT_DATASET + '/UCF_Crime_other/file_list_cv/ucf101_flow_train_split_1.txt'
        filename_imglist_val =  ROOT_DATASET + '/UCF_Crime_other/file_list_cv/ucf101_flow_val_split_1.txt'
        prefix = 'flow_{}_{:05d}.jpg'
    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix

def return_ucf101(modality):
    filename_categories = 'E://GSM//classInd.txt'
    if modality == 'RGB':
        root_data = ROOT_DATASET + 'UCF101//ucf101_jpegs_256//jpegs_256'
        filename_imglist_train = 'E://GSM//train_rgb_split1.txt'
        filename_imglist_val = 'E://GSM//val_rgb_split1.txt'
        prefix = 'frame{:06d}.jpg'
    elif modality == 'Flow':
        root_data = ROOT_DATASET + 'UCF101//ucf101_tvl1_flow//tvl1_flow'
        filename_imglist_train = 'E://GSM//train_rgb_split1.txt'
        filename_imglist_val = 'E://GSM//val_rgb_split1.txt'
        prefix = 'frame{:06d}.jpg'
    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_hmdb51(modality):
    filename_categories = 51
    if modality == 'RGB':
        root_data = ROOT_DATASET + 'HMDB51//hmdb51_jpegs_256//jpegs_256'
        filename_imglist_train = 'E://GSM//hmdb51_rgb_train_split1.txt'
        filename_imglist_val = 'E://GSM//hmdb51_rgb_val_split_1.txt'
        prefix = 'frame{:06d}.jpg'
    elif modality == 'Flow':
        root_data = ROOT_DATASET + 'HMDB51//hmdb51_tvl1_flow//tvl1_flow'
        filename_imglist_train ='E://GSM//hmdb51_rgb_train_split1.txt'
        filename_imglist_val = 'E://GSM//hmdb51_rgb_val_split_1.txt'
        prefix = 'frame{:06d}.jpg'
    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_something(modality):
    filename_categories = 'something/v1/category.txt'
    if modality == 'RGB':
        root_data = ROOT_DATASET + 'something/v1/20bn-something-something-v1'
        filename_imglist_train = 'something/v1/train_videofolder.txt'
        filename_imglist_val = 'something/v1/val_videofolder.txt'
        prefix = '{:05d}.jpg'
    elif modality == 'Flow':
        root_data = ROOT_DATASET + 'something/v1/20bn-something-something-v1-flow'
        filename_imglist_train = 'something/v1/train_videofolder_flow.txt'
        filename_imglist_val = 'something/v1/val_videofolder_flow.txt'
        prefix = '{:06d}-{}_{:05d}.jpg'
    else:
        print('no such modality:'+modality)
        raise NotImplementedError
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_somethingv2(modality):
    filename_categories = 'something/v2/category.txt'
    if modality == 'RGB':
        root_data = ROOT_DATASET + 'something/v2/20bn-something-something-v2-frames'
        filename_imglist_train = 'something/v2/train_videofolder.txt'
        filename_imglist_val = 'something/v2/val_videofolder.txt'
        prefix = '{:06d}.jpg'
    elif modality == 'Flow':
        root_data = ROOT_DATASET + 'something/v2/20bn-something-something-v2-flow'
        filename_imglist_train = 'something/v2/train_videofolder_flow.txt'
        filename_imglist_val = 'something/v2/val_videofolder_flow.txt'
        prefix = '{:06d}.jpg'
    else:
        raise NotImplementedError('no such modality:'+modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_jester(modality):
    filename_categories = 'jester/category.txt'
    if modality == 'RGB':
        prefix = '{:05d}.jpg'
        root_data = ROOT_DATASET + 'jester/20bn-jester-v1'
        filename_imglist_train = 'jester/train_videofolder.txt'
        filename_imglist_val = 'jester/val_videofolder.txt'
    else:
        raise NotImplementedError('no such modality:'+modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_kinetics(modality):
    filename_categories = 400
    if modality == 'RGB':
        root_data = ROOT_DATASET + 'kinetics/images'
        filename_imglist_train = 'kinetics/labels/train_videofolder.txt'
        filename_imglist_val = 'kinetics/labels/val_videofolder.txt'
        prefix = 'img_{:05d}.jpg'
    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_dataset(dataset, modality):
    dict_single = {'jester': return_jester, 'something': return_something, 'somethingv2': return_somethingv2,
                   'ucf101': return_ucf101, 'hmdb51': return_hmdb51, 'ucfcrime': return_ucfcrime,
                   'kinetics': return_kinetics }
    if dataset in dict_single:
        file_categories, file_imglist_train, file_imglist_val, root_data, prefix = dict_single[dataset](modality)
    else:
        raise ValueError('Unknown dataset '+dataset)

    file_imglist_train = os.path.join(ROOT_DATASET, file_imglist_train)
    file_imglist_val = os.path.join(ROOT_DATASET, file_imglist_val)
    if isinstance(file_categories, str): #isinstance() 函数来判断一个对象是否是一个已知的类型
        file_categories = os.path.join(ROOT_DATASET, file_categories)
        with open(file_categories) as f:
            lines = f.readlines()
        categories = [item.rstrip() for item in lines] # rstrip() 删除 string 字符串末尾的指定字符
    else:  # number of categories
        categories = [None] * file_categories
    n_class = len(categories)
    print('{}: {} classes'.format(dataset, n_class))
    return n_class, file_imglist_train, file_imglist_val, root_data, prefix
