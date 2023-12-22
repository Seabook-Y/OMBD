import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

class OMBD_tie(nn.Module):
    def __init__(self, ch_in, ch_out, device, KF_path):
        super(OMBD_tie, self).__init__()
        chennel = 1024
        self.conv1_3_Ek = nn.Conv1d(ch_in, chennel, kernel_size=1, stride=1, padding=0)
        self.conv1_3_Ev = nn.Conv1d(ch_in, chennel, kernel_size=1, stride=1, padding=0)
        self.conv1_3_k = nn.Conv1d(ch_in, chennel, kernel_size=1, stride=1, padding=0)
        self.conv1_3_v = nn.Conv1d(ch_in, chennel, kernel_size=1, stride=1, padding=0)
        self.conv2_1_W = nn.Conv1d(chennel, 1, kernel_size=1, stride=1, padding=0)
        self.conv2_1 = nn.Conv1d(chennel * 2, 9, kernel_size=1, stride=1, padding=0)
        self.opt = nn.ReLU()
        self.tie_feature = list()

        tie = np.load(KF_path, allow_pickle=True)

        for i in range(0, 9, 1):
            x = np.asarray(tie[i])
            x = torch.from_numpy(x).squeeze().unsqueeze(0)
            self.tie_feature.append(x.permute(0, 2, 1))
            self.tie_feature[i] = self.tie_feature[i].to(device)

    def weight(self, value, y_last):
        y_weight = torch.cosine_similarity(value, y_last, dim=1)
        y_weight = F.softmax(y_weight, dim=-1)
        y_weight = y_weight.unsqueeze(1)
        return y_weight

    def sum(self, value, y_weight):
        y_weight = y_weight.permute(0, 2, 1)
        y_sum = torch.matmul(value, y_weight)
        return y_sum

    def forward(self, x, device):
        x = x[:, -1:, :]
        x = x.permute(0, 2, 1)
        k = self.conv1_3_k(x)
        v = self.conv1_3_v(x)

        feature_w = torch.empty(x.shape[0], 9).to(device)
        for i in range(0, 9, 1):
            tie_feature = self.tie_feature[i]

            Ek = self.conv1_3_Ek(tie_feature)
            Ev = self.conv1_3_Ev(tie_feature)

            weight = self.weight(Ek, k)
            sum = self.sum(Ev, weight)
            if i == 0:
                feature_E = sum
            else:
                feature_E = torch.cat((feature_E, sum), dim=-1)

            feature_w[:, i:i + 1] = self.conv2_1_W(sum).squeeze(-1)

        feature_E = feature_E.to(device)
        feature_w = F.softmax(feature_w, dim=-1).unsqueeze(-1)
        feature_E = torch.bmm(feature_E, feature_w)
        out = torch.cat((v, feature_E), dim=1)
        out = self.opt(out)
        out = self.conv2_1(out)
        return out


class OMBD_lstra(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(OMBD_lstra, self).__init__()
        chennel = 1024

        chennel2=512
        in_channel=6144


        self.conv1_3_d1 = nn.Conv1d(ch_in, chennel, kernel_size=3, stride=1, padding=0, dilation = 1)
        self.conv1_3_d2 = nn.Conv1d(ch_in, chennel, kernel_size=3, stride=1, padding=0, dilation = 2)
        self.conv1_3_d4 = nn.Conv1d(ch_in, chennel, kernel_size=3, stride=1, padding=0, dilation = 4)
        self.conv1_3_d8 = nn.Conv1d(ch_in, chennel, kernel_size=3, stride=1, padding=0, dilation=8)

        self.conv = nn.Conv1d(ch_in, chennel, kernel_size=1, stride=1, padding=0)

        self.conv1_3_kd = nn.Conv1d(in_channel, chennel, kernel_size=1, stride=1, padding=0)
        self.conv1_3_vd = nn.Conv1d(in_channel, chennel, kernel_size=1, stride=1, padding=0)




        self.conv1_3_k = nn.Conv1d(ch_in, chennel, kernel_size=1, stride=1, padding=0)
        self.conv1_3_v = nn.Conv1d(ch_in, chennel, kernel_size=1, stride=1, padding=0)
        self.conv1_feature = nn.Conv1d(ch_in, chennel, kernel_size=3, stride=1, padding=1)
        self.conv2_3_k = nn.Conv1d(chennel, chennel, kernel_size=3, stride=1, padding=1)
        self.conv2_3_v = nn.Conv1d(chennel, chennel, kernel_size=3, stride=1, padding=1)
        self.conv3_1 = nn.Conv1d(chennel, 9, kernel_size=1, stride=1, padding=0)
        self.opt = nn.ReLU()

    def weight(self, value, y_last):
        y_weight = torch.cosine_similarity(value, y_last, dim=1)
        y_weight = F.softmax(y_weight, dim=-1)
        y_weight = y_weight.unsqueeze(1)
        return y_weight

    def sum(self, value, y_weight):
        y_weight = y_weight.permute(0, 2, 1)
        y_sum = torch.bmm(value, y_weight)
        sum = value[:, :, -1:] + y_sum
        return torch.cat((value[:, :, :-1], sum), dim=-1)

    def forward(self, input):
        input = input.permute(0, 2, 1)

        pad1 = nn.ZeroPad2d(padding=(2,0,0,0))
        input1=pad1(input)

        pad2 = nn.ZeroPad2d(padding=(4, 0, 0, 0))
        input2 = pad2(input)

        pad4 = nn.ZeroPad2d(padding=(8, 0, 0, 0))
        input4 = pad4(input)

        pad8 = nn.ZeroPad2d(padding=(16, 0, 0, 0))
        input8 = pad8(input)
        #
        d1 = self.conv1_3_d1(input1)
        d2 = self.conv1_3_d2(input2)
        d4 = self.conv1_3_d4(input4)
        d8 = self.conv1_3_d8(input8)
        #
        C1 = torch.cat( (d1,d2), 1 )
        C2 = torch.cat((d4, d8), 1)
        C= torch.cat((C1, C2), 1)
        CI= torch.cat((C, input), 1)


        k = self.conv1_3_kd(CI)
        v = self.conv1_3_vd(CI)

        y_weight = self.weight(k, k[:, :, -1:])
        feat1 = self.sum(v, y_weight)
        feat1 = self.opt(feat1)

        k = self.conv2_3_k(feat1)
        v = self.conv2_3_v(feat1)
        y_weight = self.weight(k, k[:, :, -1:])
        feat2 = self.sum(v, y_weight)
        feat2 = self.opt(feat2)

        return self.conv3_1(feat2)


if __name__ == '__main__':
    import time
    
    lstra_input = torch.randn(1, 64, 2048)
    tie_input = torch.randn(1, 1, 2048)
    device = torch.device("cuda:0")
    lstra_input = lstra_input.to(device)
    tie_input = tie_input.to(device)

    model_d = OMBD_lstra(2048, 9)
    model_s = OMBD_tie(2048, 9, device)
    model_d.to(device)
    model_s.to(device)

    start = time.time()
    for i in range(290):
        out1 = model_d(lstra_input)
        out2 = model_s(tie_input, device)
    print(time.time() - start)
