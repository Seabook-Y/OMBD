import argparse
import pickle
import numpy as np

import copy
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def get_args():
    parser = argparse.ArgumentParser(description='Generating category KFs with LDA.')
    parser.add_argument('--feature_file', default='./data/PDMB_train.pickle')
    parser.add_argument('--anno_file', default='./data/PDMB_train_anno.pickle')
    parser.add_argument('--LDA_num', default=9)
    parser.add_argument('--category_num', default=9)
    parser.add_argument('--KF_file', default='./data/KF.pickle')
    args = parser.parse_args()
    return args


def select_feature(args):
    targets = pickle.load(open(args.anno_file, 'rb'))
    features = pickle.load(open(args.feature_file, 'rb'))

    # store all features
    features_all = [None for _ in range(args.category_num)]

    for vid_name, annos in targets.items():
        vid_anno = annos['anno']
        feat_rgb = features[vid_name]['rgb']
        feat_flow = features[vid_name]['flow']
        feature = np.concatenate([feat_rgb, feat_flow], axis=1)


        category_indicator = np.sum(vid_anno, axis=0)
        for i in range(args.category_num):
            if category_indicator[i] == 0:
                continue

            cate_flag = vid_anno[:, i]
            places = np.where(cate_flag == 1)
            cate_feat = feature[places[0], :]
            tmp = features_all[i]
            if tmp is None:
                tmp = cate_feat
            else:
                tmp = np.concatenate([tmp, cate_feat], axis=0)
            features_all[i] = tmp


    return features_all


def KFs_LDA(args, features_all):
    KFs = list()

    num_f=0
    for cate_feat in features_all:

        features_all_2= copy.deepcopy(features_all)

        l1=np.zeros(cate_feat.shape[0])

        del features_all_2[num_f]
        num_f=num_f+1
        features_all_3 = []

        for num2 in range(len(features_all_2)):
             if num2==0:
                 features_all_3=features_all_2[0]
             else:
                 features_all_3=np.concatenate((features_all_3,features_all_2[num2]),axis=0)
        l2=np.ones(features_all_3.shape[0])
        label= np.concatenate((l1,l2),axis=0)
        features_all_4 = np.concatenate((cate_feat, features_all_3), axis=0)
        lda=LinearDiscriminantAnalysis()
        feature=lda.fit_transform(features_all_4,label)
        feature_T=np.transpose(feature)
        feature_T_2=feature_T[:, :cate_feat.shape[0]]
        extracted_feature=np.dot(feature_T_2,cate_feat)
        d=cate_feat- extracted_feature
        d_2=np.linalg.norm(d,ord=2,axis=1)
        cate_KF = list()
        idx = np.argsort(d_2)[:args.LDA_num]
        for i in range(args.LDA_num):
            cate_KF.append(cate_feat[idx[i], :])
        KFs.append(cate_KF)
        print(np.array(KFs).shape)

    pickle.dump(KFs, open(args.KF_file, 'wb'))
    return


if __name__ == '__main__':
    args = get_args()
    features_all = select_feature(args)
    KFs_LDA(args, features_all)
