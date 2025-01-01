import torch
import torch.nn as nn
import torch.nn.functional as F
from resnet import Bottleneck

class AttentionNet(nn.Module):
    def __init__(self, backbone, ft_flag, img_size, hid_dim, c, w, h,
                 attritube_num, cls_num, ucls_num, attr_group, w2v,
                 scale=20.0, device=None):
        super(AttentionNet, self).__init__()
        
        self.hid_dim = 0# hid_dim
        # self.prototype_shape = prototype_shape
        self.device = device

        self.name = "AttentionNet"

        self.img_size = img_size
        # self.prototype_shape = prototype_shape
        self.attritube_num = attritube_num

        self.feat_channel = c
        self.feat_w = w
        self.feat_h = h
        self.feat_n = w * h

        self.ucls_num = ucls_num
        self.scls_num = cls_num - ucls_num
        self.attr_group = attr_group
        # global branch
        print(w2v.shape)
        self.w2v_att = torch.from_numpy(w2v).float().to(self.device)  # 312 * 300
        assert self.w2v_att.shape[0] == self.attritube_num
        _, self.w2v_length = self.w2v_att.shape

        if scale <= 0:
            self.scale1 = torch.ones(1) * 20.0
            self.scale2 = torch.ones(1) * 20.0
            self.scale3 = torch.ones(1) * 20.0
        else:
            self.scale1 = torch.tensor(scale)
            self.scale2 = torch.tensor(scale)
            self.scale3 = torch.tensor(scale)

        self.backbone = backbone  # requires_grad = True

        self.ft_flag = ft_flag
        self.check_fine_tune()

        self.med_dim = 1024  # 1024
        # local branch 1
        local_feat_channel = 2048
        self.QueryW1 = nn.Linear(self.w2v_length, self.med_dim)  # L,M = 300,1024
        self.KeyW1 = nn.Linear(local_feat_channel, self.med_dim)  # C,M = 2048,1024
        self.ValueW1 = nn.Linear(local_feat_channel, self.med_dim)  # C,M = 2048,1024
        self.W_o1 = nn.Linear(self.med_dim, self.feat_channel)  # M,C = 1024,2048
        self.V_att_final_branch1 = nn.Parameter(nn.init.normal_(torch.empty(self.attritube_num, self.feat_channel)),
                                                requires_grad=True)  # S, C
        # local branch 2
        local_feat_channel = 2048
        self.QueryW2 = nn.Linear(self.w2v_length, self.med_dim)  # L,M = 300,1024
        self.KeyW2 = nn.Linear(local_feat_channel, self.med_dim)  # C,M = 2048,1024
        self.ValueW2 = nn.Linear(local_feat_channel, self.med_dim)  # C,M = 2048,1024
        self.W_o2 = nn.Linear(self.med_dim, self.feat_channel)  # M,C = 1024,2048
        self.V_att_final_branch2 = nn.Parameter(nn.init.normal_(torch.empty(self.attritube_num, self.feat_channel)),
                                                requires_grad=True)  # S, C
        # local branch 3
        local_feat_channel = 2048
        self.QueryW3 = nn.Linear(self.w2v_length, self.med_dim)  # L,M = 300,1024
        self.KeyW3 = nn.Linear(local_feat_channel, self.med_dim)  # C,M = 2048,1024
        self.ValueW3 = nn.Linear(local_feat_channel, self.med_dim)  # C,M = 2048,1024
        self.W_o3 = nn.Linear(self.med_dim, self.feat_channel)  # M,C = 1024,2048
        self.V_att_final_branch3 = nn.Parameter(nn.init.normal_(torch.empty(self.attritube_num, self.feat_channel)),
                                                requires_grad=True)  # S, C
        
        self.s5 = RFM_Module(local_feat_channel, local_feat_channel, n_upsamples=0)
        self.s4 = RFM_Module(local_feat_channel, local_feat_channel, n_upsamples=1)#(local_feat_channel//2, local_feat_channel, n_upsamples=1)
        self.s3 = RFM_Module(local_feat_channel//2, local_feat_channel, n_upsamples=1)#(local_feat_channel//4, local_feat_channel, n_upsamples=2)
        
        self.lateral_conv1 = SCAM(self.feat_channel*2, self.feat_channel)
        self.lateral_conv2 = SCAM(self.feat_channel, self.feat_channel)

    def forward(self, x, label_att=None, label=None, support_att=None, getAttention = False):
        feat1, feat2, feat3 = self.conv_features(x)
        if getAttention:
            pass
        else:
            v2s1 = self.res_attention_module1([feat1, self.lateral_conv1(torch.cat([feat2.detach(), feat3.detach()], dim=1))], support_att)  # B,312
            v2s2 = self.res_attention_module2([feat2, self.lateral_conv2(feat3.detach())], support_att)  # B,312
            v2s3 = self.res_attention_module3([feat3], support_att)  # B,312

            return v2s1, v2s2, v2s3
    
    def res_attention_module1(self, feats, s, getAttention = False):
        """
        feat: [B, C, W, H]
        """
        feat_reshape = []
        for feat in feats:
            B, C, W, H = feat.shape
            N = W * H
            S, L = self.w2v_att.shape
            M = self.med_dim
            W = H = int(N ** 0.5)
            # attention feature
            feat_reshape.append(feat.reshape(B, C, W * H))  # B, C, N=WH
        feat_reshape = torch.cat(feat_reshape, dim=2)
        w2v_att = self.w2v_att.to(self.device)
        query = self.QueryW1(w2v_att) # [S, L]*[L,M] -> [S,M]
        query_batch = query.unsqueeze(0).repeat(B, 1, 1)  # [S,M] -> [1,S,M] -> [B,S,M]
        key = self.KeyW1(feat_reshape.permute(0, 2, 1))  # [B, C, N] -> [B, N, C] -> [B, N, M]
        value = self.ValueW1(feat_reshape.permute(0, 2, 1))  # [B, C, N] -> [B, N, C] -> [B, N, M]

        attention = F.softmax(torch.matmul(query_batch, key.permute(0, 2, 1)),
                              dim=2)  # [B,S,M],[B,N,M] -> [B,S,M],[B,M,N] -> [B, S, N]
        attented_feat = torch.matmul(attention, value)  # [B, S, N] * [B, N, M] -> [B,S,M]
        attented_feat_final = self.W_o1(attented_feat)  # [B,S,M] -> [B,S,C]

        # visual to semantic
        if self.hid_dim == 0:
            v2s = torch.einsum('BSC,SC->BS', attented_feat_final, self.V_att_final_branch1)  # [B,312,2048] * [312, 2048] -> [B,312,2048] -> [B,312]
        else:
            attented_feat_hid = self.V_att_hidden_branch1(attented_feat_final) # [B,312,2048] -> [B,312,4096]
            v2s = torch.einsum('BSH,SH->BS', attented_feat_hid, self.V_att_final_branch1)

        if getAttention:
            return v2s, attention
        else:
            return v2s

    def res_attention_module2(self, feats, s, getAttention = False):
        """
        feat: [B, C, W, H]
        """
        feat_reshape = []
        for feat in feats:
            B, C, W, H = feat.shape
            N = W * H
            S, L = self.w2v_att.shape
            M = self.med_dim
            W = H = int(N ** 0.5)
            # attention feature
            feat_reshape.append(feat.reshape(B, C, W * H))  # B, C, N=WH
        feat_reshape = torch.cat(feat_reshape, dim=2)
        w2v_att = self.w2v_att.to(self.device)
        query = self.QueryW2(w2v_att) # [S, L]*[L,M] -> [S,M]
        query_batch = query.unsqueeze(0).repeat(B, 1, 1)  # [S,M] -> [1,S,M] -> [B,S,M]
        key = self.KeyW2(feat_reshape.permute(0, 2, 1))  # [B, C, N] -> [B, N, C] -> [B, N, M]
        value = self.ValueW2(feat_reshape.permute(0, 2, 1))  # [B, C, N] -> [B, N, C] -> [B, N, M]

        attention = F.softmax(torch.matmul(query_batch, key.permute(0, 2, 1)),
                              dim=2)  # [B,S,M],[B,N,M] -> [B,S,M],[B,M,N] -> [B, S, N]
        attented_feat = torch.matmul(attention, value)  # [B, S, N] * [B, N, M] -> [B,S,M]
        attented_feat_final = self.W_o2(attented_feat)  # [B,S,M] -> [B,S,C]

        # visual to semantic
        if self.hid_dim == 0:
            v2s = torch.einsum('BSC,SC->BS', attented_feat_final, self.V_att_final_branch2)  # [B,312,2048] * [312, 2048] -> [B,312,2048] -> [B,312]
        else:
            attented_feat_hid = self.V_att_hidden_branch2(attented_feat_final) # [B,312,2048] -> [B,312,4096]
            v2s = torch.einsum('BSH,SH->BS', attented_feat_hid, self.V_att_final_branch2)
        
        if getAttention:
            return v2s, attention
        else:
            return v2s

    def res_attention_module3(self, feats, s, getAttention = False):
        """
        feat: [B, C, W, H]
        """
        feat_reshape = []
        for feat in feats:
            B, C, W, H = feat.shape
            N = W * H
            S, L = self.w2v_att.shape
            M = self.med_dim
            W = H = int(N ** 0.5)
            # attention feature
            feat_reshape.append(feat.reshape(B, C, W * H))  # B, C, N=WH
        feat_reshape = torch.cat(feat_reshape, dim=2)
        w2v_att = self.w2v_att.to(self.device)
        query = self.QueryW3(w2v_att) # [S, L]*[L,M] -> [S,M]
        query_batch = query.unsqueeze(0).repeat(B, 1, 1)  # [S,M] -> [1,S,M] -> [B,S,M]
        key = self.KeyW3(feat_reshape.permute(0, 2, 1))  # [B, C, N] -> [B, N, C] -> [B, N, M]
        value = self.ValueW3(feat_reshape.permute(0, 2, 1))  # [B, C, N] -> [B, N, C] -> [B, N, M]

        attention = F.softmax(torch.matmul(query_batch, key.permute(0, 2, 1)),
                              dim=2)  # [B,S,M],[B,N,M] -> [B,S,M],[B,M,N] -> [B, S, N]
        attented_feat = torch.matmul(attention, value)  # [B, S, N] * [B, N, M] -> [B,S,M]
        attented_feat_final = self.W_o3(attented_feat)  # [B,S,M] -> [B,S,C]

        # visual to semantic
        if self.hid_dim == 0:
            v2s = torch.einsum('BSC,SC->BS', attented_feat_final, self.V_att_final_branch3)  # [B,312,2048] * [312, 2048] -> [B,312,2048] -> [B,312]
        else:
            attented_feat_hid = self.V_att_hidden_branch3(attented_feat_final) # [B,312,2048] -> [B,312,4096]
            v2s = torch.einsum('BSH,SH->BS', attented_feat_hid, self.V_att_final_branch3)
        
        if getAttention:
            return v2s, attention
        else:
            return v2s

    def conv_features(self, x):
        '''
        the feature input to prototype layer
        '''
        _, _, c3, c4, c5 = self.backbone(x)

        s5 = self.s5(c5)
        s4 = self.s4(c4)
        s3 = self.s3(c3)

        return s3, s4, s5

    def euclidean_dist(self, prediction, support_att, norm=False):
        if norm == False:
            N, S = prediction.shape
            C, S = support_att.shape

            support_att_expand = support_att.unsqueeze(0).expand(N, C, S)
            prediction_expand = prediction.unsqueeze(1).expand(N, C, S)
            offset = torch.sum((prediction_expand - support_att_expand) ** 2, dim=2)  # [N, C, S]-->[N,C]
            return offset
        else:
            N, S = prediction.shape
            C, S = support_att.shape
            support_att_norm = torch.norm(support_att, p=2, dim=1).unsqueeze(1).expand_as(support_att)
            support_att_normalized = support_att.div(support_att_norm + 1e-10)
            prediction_norm = torch.norm(prediction, p=2, dim=1).unsqueeze(1).expand_as(prediction)
            prediction_normalized = prediction.div(prediction_norm + 1e-10)

            support_att_expand = support_att_normalized.unsqueeze(0).expand(N, C, S)
            prediction_expand = prediction_normalized.unsqueeze(1).expand(N, C, S)
            offset = torch.sum((prediction_expand - support_att_expand) ** 2, dim=2)  # [N, C, S]-->[N,C]

            return offset

    def cosine_dis(self, pred_att, support_att, stage='1'):
        pred_att_norm = torch.norm(pred_att, p=2, dim=1).unsqueeze(1).expand_as(pred_att)
        pred_att_normalized = pred_att.div(pred_att_norm + 1e-10)
        support_att_norm = torch.norm(support_att, p=2, dim=1).unsqueeze(1).expand_as(support_att)
        support_att_normalized = support_att.div(support_att_norm + 1e-10)
        cos_dist = torch.einsum('bd,nd->bn', pred_att_normalized, support_att_normalized)
        if stage == '1':
            score = cos_dist * self.scale1  # B, cls_num
        elif stage == '2':
            score = cos_dist * self.scale2  # B, cls_num
        elif stage == '3':
            score = cos_dist * self.scale3  # B, cls_num
        return score, cos_dist

    def check_fine_tune(self):
        if self.ft_flag:
            for p in self.backbone.parameters():
                p.requires_grad = True
        else:
            for p in self.backbone.parameters():
                p.requires_grad = False

class Conv3x3GNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, upsample=False):
        super().__init__()
        self.upsample = upsample
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (3, 3),
                      stride=1, padding=1, bias=False),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.block(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=True)
        return x

class SegmentationBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_upsamples=0):
        super().__init__()

        blocks = [
            Conv3x3GNReLU(in_channels, out_channels, upsample=bool(n_upsamples))
        ]

        if n_upsamples > 1:
            for _ in range(1, n_upsamples):
                blocks.append(Conv3x3GNReLU(out_channels, out_channels, upsample=True))

        self.block = nn.Sequential(*blocks)

    def forward(self, x):
        return self.block(x)

class RFM_Module(nn.Module):
    def __init__(self, in_dim, out_dim, n_upsamples):
        super(RFM_Module, self).__init__()
        from grouping import GroupingUnit
        self.group = GroupingUnit(in_dim, 5)
        self.group.reset_parameters(init_weight=None, init_smooth_factor=None)
        self.conv = SegmentationBlock(in_dim, out_dim, n_upsamples)

    def forward(self, x):
        x = self.group(x)
        x = self.conv(x)
        
        return x

class SCAM(nn.Module):
    """ Spatial self-attention block """
    def __init__(self, in_chan, out_chan):
        super(SCAM, self).__init__()
        self.conv = nn.Conv2d(in_chan, out_chan, kernel_size=1, bias=False)
        self.c_attn = nn.Sequential(nn.Linear(out_chan, out_chan),
                                     nn.ReLU(),
                                     nn.Linear(out_chan, out_chan),
                                     nn.Sigmoid())

        self.s_attn = nn.Sequential(nn.Conv2d(out_chan, out_chan, 3, padding=1, bias=False),
                                        nn.Sigmoid())

        import fvcore.nn.weight_init as weight_init
        weight_init.c2_xavier_fill(self.conv)
    
    def forward(self, x):
        x = self.conv(x)

        # Channel
        B, C, H, W = x.size()
        c_attn = self.c_attn(torch.mean(x, dim=(2,3))).view(B,C,1,1)

        # spatial
        s_attn = self.s_attn(x)
        
        return torch.mul(x, c_attn*s_attn)