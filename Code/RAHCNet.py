import torch.nn as nn
import torch
import torch.nn.functional as F


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        # Post-activation
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5,
                                 momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        # x = self.conv(x)
        return x


class CrossAttnFusion(nn.Module):
    def __init__(self, dim=512, heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, vis, txt):
        # vis, txt: [B, 512]
        q = vis.unsqueeze(1)
        k = txt.unsqueeze(1)
        v = txt.unsqueeze(1)

        out, _ = self.attn(q, k, v)
        out = out.squeeze(1)
        return self.norm(vis + out)


# class BiCrossAttn(nn.Module):
#     def __init__(self, dim=512, heads=8):
#         super().__init__()
#         self.cross_attn = CrossAttnFusion(dim, heads)
#
#     def forward(self, vis, txt):
#         v2t = self.cross_attn(vis, txt)
#         t2v = self.cross_attn(txt, vis)
#         return (v2t + t2v) / 2


class RAHC(nn.Module):
    def __init__(self, model, feature_size, dataset):
        super(RAHC, self).__init__()

        self.features = nn.Sequential(*list(model.children())[:-2])
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU()
        self.num_ftrs = 2048 * 1 * 1

        self.conv_block1 = nn.Sequential(
            BasicConv(self.num_ftrs, feature_size, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(feature_size, self.num_ftrs, kernel_size=3, stride=1, padding=1, relu=True)
        )
        self.conv_block2 = nn.Sequential(
            BasicConv(self.num_ftrs, feature_size, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(feature_size, self.num_ftrs, kernel_size=3, stride=1, padding=1, relu=True)
        )
        self.conv_block3 = nn.Sequential(
            BasicConv(self.num_ftrs, feature_size, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(feature_size, self.num_ftrs, kernel_size=3, stride=1, padding=1, relu=True)
        )

        self.fc1 = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs),
            nn.Linear(self.num_ftrs, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Linear(feature_size, 512)
        )

        self.fc2 = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs),
            nn.Linear(self.num_ftrs, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Linear(feature_size, 512)
        )

        self.fc3 = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs),
            nn.Linear(self.num_ftrs, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Linear(feature_size, 512)
        )

        self.fc4 = nn.Sequential(
            nn.BatchNorm1d(1024),
            nn.Linear(1024, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Linear(feature_size, 512)
        )

        if dataset == 'CUB':
            self.classifier_1 = nn.Sequential(
                nn.Linear(512, 13),
                nn.Sigmoid()
            )
            self.classifier_2 = nn.Sequential(
                nn.Linear(512, 38),
                nn.Sigmoid()
            )
            self.classifier_3 = nn.Sequential(
                nn.Linear(512, 200),
                nn.Sigmoid()
            )
            self.classifier_3_1 = nn.Sequential(
                nn.Linear(512, 200)
            )
        elif dataset == 'Air':
            self.classifier_1 = nn.Sequential(
                nn.Linear(512, 30),
                nn.Sigmoid()
            )
            self.classifier_2 = nn.Sequential(
                nn.Linear(512, 70),
                nn.Sigmoid()
            )
            self.classifier_3 = nn.Sequential(
                nn.Linear(512, 100),
                nn.Sigmoid()
            )
            self.classifier_3_1 = nn.Sequential(
                nn.Linear(512, 100)
            )
        self.atten = CrossAttnFusion(dim=512, heads=8)

    # def forward(self, x):  # (8,3,448,448)
    def forward(self, x, coarse_x, middle_x, fine_x):  # (8,3,448,448)
        x = self.features(x)  # (8,2048,14,14)
        x_order = self.conv_block1(x)
        x_family = self.conv_block2(x)
        x_species = self.conv_block3(x)

        x_order_fc = self.pooling(x_order)  # torch.Size([32, 2048, 1, 1])
        x_order_fc = x_order_fc.view(x_order_fc.size(0), -1)  # torch.Size([32, 2048])
        x_order_fc = self.fc1(x_order_fc)  # (8,512)
        x_family_fc = self.pooling(x_family)
        x_family_fc = x_family_fc.view(x_family_fc.size(0), -1)
        x_family_fc = self.fc2(x_family_fc)  # (8,512)
        x_species_fc = self.pooling(x_species)
        x_species_fc = x_species_fc.view(x_species_fc.size(0), -1)
        x_species_fc = self.fc3(x_species_fc)  # (8,512)

        x_coarse_text = self.fc4(coarse_x)
        x_middle_text = self.fc4(middle_x)
        x_fine_text = self.fc4(fine_x)

        x_order_v_t = self.atten(x_order_fc, x_coarse_text)
        x_family_v_t = self.atten(x_family_fc, x_middle_text)
        x_species_v_t = self.atten(x_species_fc, x_fine_text)

        # x_order_family_v_v = self.atten(x_order_fc, x_family_fc)
        # x_family_species_v_v = self.atten(x_family_fc, x_species_fc)
        # x_order_family_t_t = self.atten(x_coarse_text, x_middle_text)
        # x_family_species_t_t = self.atten(x_middle_text, x_fine_text)

        y_order_sig = self.classifier_1(self.relu(x_order_v_t))  # coarse = self.relu(x_order_fc) (8,512)
        y_family_sig = self.classifier_2(self.relu(x_order_v_t + x_family_v_t))
        y_species_sig = self.classifier_3(self.relu(x_order_v_t + x_family_v_t + x_species_v_t))
        # y_species_sof = self.classifier_3_1(self.relu(x_species_fc + x_family_fc + x_order_fc + x_fine_text + x_middle_text + x_coarse_text))  # Lce

        y_species_sof_v = self.classifier_3_1(self.relu(x_species_fc + x_family_fc + x_order_fc))
        y_species_sof_t = self.classifier_3_1(self.relu(x_fine_text + x_middle_text + x_coarse_text))

        return y_order_sig, y_family_sig, 0, y_species_sig, y_species_sof_v, y_species_sof_t
    