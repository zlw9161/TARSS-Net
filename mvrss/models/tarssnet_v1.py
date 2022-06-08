#### zlw@20220331 ####
#### TARSS-Net_V1 ####
#### MVNet w/ temporal att. pooling w/ pos_embedding ####
#### Spatial Squeeze w/ GAP & GMP w/ PosEnc (Attention is all you need) ####
#### TAM-ASPP Fusion & Latent Space
#### ATT-Pool: weighted summation + current frame;
#### TAM: remove augmentation FC layer; using smaller TRE block; using only x as input;
#### ASPP: using x_latent as input (same with the Latent Space)
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DoubleConvBlock(nn.Module):
    """ (2D conv => BN => LeakyReLU) * 2 """

    def __init__(self, in_ch, out_ch, k_size, pad, dil):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k_size, padding=pad, dilation=dil),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=k_size, padding=pad, dilation=dil),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x = self.block(x)
        return x


class Double3DConvBlock(nn.Module):
    """ (3D conv => BN => LeakyReLU) * 2 (temporal shared 2D Conv)"""

    def __init__(self, in_ch, out_ch, k_size, strd, pad, dil):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=k_size, stride=strd, padding=pad, dilation=dil),
            nn.BatchNorm3d(out_ch),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=k_size, stride=strd, padding=pad, dilation=dil),
            nn.BatchNorm3d(out_ch),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x = self.block(x)
        return x


class ConvBlock(nn.Module):
    """ (2D conv => BN => LeakyReLU) """

    def __init__(self, in_ch, out_ch, k_size, pad, dil):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k_size, padding=pad, dilation=dil),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x = self.block(x)
        return x

class TREBlock(nn.Module):
    """ (3D conv => BN => LeakyReLU) """
    """ Temporal Relation Embeddings """

    def __init__(self, in_ch, out_ch, k_size, strd, pad):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=k_size, stride=strd, padding=pad),
            nn.BatchNorm3d(out_ch),
            nn.LeakyReLU(inplace=True))

    def forward(self, x):
        x = self.block(x)
        return x

'''
class PosEnc(nn.Module):
    """ Temporal Positional Encoding """

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pos_vec =  torch.tensor([[0.2000, 1.0000, 0.8000, 0.8000],
                                      [0.4000, 1.0000, 0.6000, 0.6000],
                                      [0.6000, 1.0000, 0.4000, 0.4000],
                                      [0.8000, 1.0000, 0.2000, 0.2000],
                                      [1.0000, 1.0000, 0.0000, 0.0000]])

    def forward(self, batch_size):
        pos_tensor = torch.unsqueeze(self.pos_vec, 0)
        pos_tensor = torch.repeat_interleave(pos_tensor, repeats=batch_size, dim=0)
        #pos_enc = self.fc_layer(pos_tensor)
        return pos_tensor
'''

class PosEnc(nn.Module):
    """ Positional Encoding using "Attention is All U Need" version """
    """ zlw@20220215 """

    def __init__(self, coef=0.1, const=8):
        super().__init__()
        self.Coef = coef
        self.Const = const

    def forward(self, batch_size, nfrms, chns):
        pos_vec = torch.zeros(nfrms, chns)
        weight = torch.ones(1)
        for i in range(int(chns/2) - 1):
            for t in range(nfrms):
                weight = weight * t / pow(10, self.Const*i/chns)
                pos_vec[t][2*i] = self.Coef * torch.sin(weight)
                pos_vec[t][2*i+1] = self.Coef * torch.cos(weight)
        pos_tensor = torch.unsqueeze(pos_vec, 0)
        pos_tensor = torch.repeat_interleave(pos_tensor, repeats=batch_size, dim=0)
        return pos_tensor

class ATTPool(nn.Module):
    """ Attentive Pooling """

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pos_enc = PosEnc(0.1, 8)
        self.fc_layer = nn.Linear(in_ch, out_ch)
        self.att_scoring = nn.Softmax(dim=1)
        # self.fc_layer_pos = nn.Linear(4, 8)
        # self.fc_layer = nn.Linear(in_ch+8, out_ch)
        # self.pos_enc = PosEnc(4, 8)

    def forward(self, x, x_tre):
        """ x_tre: B x T x C, x: B x C x T x H x W, x_pos: B x T x 8 """
        """ x_tre w/ positional encoding """

        # x_pos = self.pos_enc(x_tre.shape[0])
        x_pos = self.pos_enc(x_tre.shape[0], x_tre.shape[1], x_tre.shape[2])
        x_pos = x_pos.to(x_tre.device)
        # x_pos = self.fc_layer_pos(x_pos)
        # x_tre = torch.cat((x_tre, x_pos), 2)
        # x_tre = torch.cat((x_tre, x_pos), 2)
        x_tre = torch.add(x_tre, x_pos) # 20220222
        x_att = self.fc_layer(x_tre)
        # generate time-wise attention scores
        att_sc = self.att_scoring(x_att)
        # reshape the att_sc and broadcast each score as a vector
        att_sc = torch.permute(att_sc, (0,2,1))
        att_sc = torch.unsqueeze(att_sc, 3)
        att_sc = torch.unsqueeze(att_sc, 4)
        att_sc = torch.repeat_interleave(att_sc, repeats=x.shape[4], dim=3)
        att_sc = torch.repeat_interleave(att_sc, repeats=x.shape[4], dim=4)
        x_att = att_sc * x
        # sum along the time dimension
        x_att = torch.sum(x_att, dim=2)
        x_att = x_att + x[:, :, -1, :, :]
        return x_att

class ASPPBlock(nn.Module):
    """Atrous Spatial Pyramid Pooling
    Parallel conv blocks with different dilation rate
    """

    def __init__(self, in_ch, out_ch=256):
        super().__init__()
        self.global_avg_pool = nn.AvgPool2d((64, 64))
        self.conv1_1x1 = nn.Conv2d(in_ch, out_ch, kernel_size=1, padding=0, dilation=1)
        self.single_conv_block1_1x1 = ConvBlock(in_ch, out_ch, k_size=1, pad=0, dil=1)
        self.single_conv_block1_3x3 = ConvBlock(in_ch, out_ch, k_size=3, pad=6, dil=6)
        self.single_conv_block2_3x3 = ConvBlock(in_ch, out_ch, k_size=3, pad=12, dil=12)
        self.single_conv_block3_3x3 = ConvBlock(in_ch, out_ch, k_size=3, pad=18, dil=18)

    def forward(self, x):
        x1 = F.interpolate(self.global_avg_pool(x), size=(64, 64), align_corners=False,
                           mode='bilinear')
        x1 = self.conv1_1x1(x1)
        x2 = self.single_conv_block1_1x1(x)
        x3 = self.single_conv_block1_3x3(x)
        x4 = self.single_conv_block2_3x3(x)
        x5 = self.single_conv_block3_3x3(x)
        x_cat = torch.cat((x2, x3, x4, x5, x1), 1)
        return x_cat

class TAM(nn.Module):
    """Attention-based Temporal-aware module
    mlp_embedding + mlp_att_weight + attentive pooling
    input: _, x2_down = EncodingBranch(x) CxTxHxW tensor
    output: CxHxW tensor
    """

    def __init__(self, in_ch, out_ch, signal_type):
        super().__init__()
        self.signal_type = signal_type
        # temporal relation embedding
        self.tr_embedding_1 = TREBlock(in_ch*2, out_ch, 
                                       k_size=(1, 3, 3),
                                       strd=(1, 2, 2),
                                       pad=(0, 1, 1))
        self.tr_embedding_2 = TREBlock(out_ch, out_ch, 
                                       k_size=(1, 3, 3),
                                       strd=(1, 2, 2),
                                       pad=(0, 1, 1))
        '''self.tr_embedding_3 = TREBlock(in_ch*2, out_ch*2, 
                                       k_size=(1, 2, 2),
                                       strd=(1, 2, 2),
                                       pad=(0, 0, 0))'''
        self.spatial_squeeze_avg = nn.AvgPool3d(kernel_size=(1, 16, 16),
                                                stride=(1, 16, 16))
        self.spatial_squeeze_max = nn.MaxPool3d(kernel_size=(1, 16, 16),
                                                stride=(1, 16, 16))
        self.fc_layer_avg_max = nn.Linear(out_ch*2, out_ch)
        # initialize attention matrix
        # self.att_pool = nn.Parameter(torch.Tensor(out_ch*4, 1))
        self.att_pool = ATTPool(out_ch, out_ch)

        # augmented embedding
        # self.single_conv_block1_1x1 = ConvBlock(out_ch*2, out_ch, k_size=1,
        #                                         pad=0, dil=1)

    def forward(self, x):
        # split the target frame tensor
        x_tgt = x[:, :, -1, :, :]
        x_tgt = torch.unsqueeze(x_tgt, 2)
        # expand x_tgt as the same shape of x_down
        x_tgt_exp = x_tgt.expand_as(x)
        # cat or minus????
        x0 = torch.cat((x_tgt_exp, x), 1)

        # temporal relation encoding
        x1 = self.tr_embedding_1(x0)
        x2 = self.tr_embedding_2(x1)
        # print("x2: ", x2.shape)
        x2_spsq_avg = self.spatial_squeeze_avg(x2)
        x2_spsq_max = self.spatial_squeeze_max(x2)
        x2_spsq = torch.cat((x2_spsq_avg, x2_spsq_max), 1)
        x_tre = torch.squeeze(x2_spsq, 3)
        x_tre = torch.squeeze(x_tre, 3)
        x_tre = x_tre.permute(0, 2, 1)
        x_tre = self.fc_layer_avg_max(x_tre)
        # attentive pooling
        x_tam = self.att_pool(x, x_tre)
        # x_cat = torch.cat((x_tam, x_tmpagg), 1)
        # x_aug = self.single_conv_block1_1x1(x_cat)
        # x_aug = x_tam + x_ltnt
        # return x_aug
        return x_tam

class EncodingBranch(nn.Module):
    """
    Backbone Encoding branch for a single radar view

    PARAMETERS
    ----------
    signal_type: str
        Type of radar view.
        Supported: 'range_doppler', 'range_angle' and 'angle_doppler'
        Tensor shape: time x range(angle) x doppler(angle)
    """

    def __init__(self, signal_type, n_frames):
        super().__init__()
        self.n_frames = n_frames
        self.signal_type = signal_type
        self.double_3dconv_block1 = Double3DConvBlock(in_ch=1, out_ch=128, 
                                                      k_size=(1, 3, 3),
                                                      strd=(1, 1, 1),
                                                      pad=(0, 1, 1), dil=1)
        self.doppler_max_pool3d = nn.MaxPool3d(kernel_size=(1, 2, 2),
                                               stride=(1, 2, 1))
        self.max_pool3d = nn.MaxPool3d(kernel_size=(1, 2, 2), 
                                       stride=(1, 2, 2))
        self.double_3dconv_block2 = Double3DConvBlock(in_ch=128, out_ch=128, 
                                                      k_size=(1, 3, 3),
                                                      strd=(1, 1, 1),
                                                      pad=(0, 1, 1), dil=1)
        # max aggregation -> avg aggregation
        self.temp_avg_pool = nn.AvgPool3d(kernel_size=(self.n_frames, 1, 1), 
                                          stride=(self.n_frames, 1, 1))
        self.single_conv_block1_1x1 = ConvBlock(in_ch=128, out_ch=128, k_size=1,
                                                pad=0, dil=1)

    def forward(self, x):
        x1 = self.double_3dconv_block1(x)

        if self.signal_type in ('range_doppler', 'angle_doppler'):
            # The Doppler dimension requires a specific processing
            # padding the last two dimentions of the tensor 
            x1_pad = F.pad(x1, (0, 1, 0, 0), "constant", 0) # F.pad(top, bottom, left, right)
            x1_down = self.doppler_max_pool3d(x1_pad)
        else:
            x1_down = self.max_pool3d(x1)

        x2 = self.double_3dconv_block2(x1_down)
        if self.signal_type in ('range_doppler', 'angle_doppler'):
            # The Doppler dimension requires a specific processing
            x2_pad = F.pad(x2, (0, 1, 0, 0), "constant", 0)
            x2_down = self.doppler_max_pool3d(x2_pad)
        else:
            x2_down = self.max_pool3d(x2)

        # temporal pooling the x2_down
        x2_tmp = self.temp_avg_pool(x2_down)
        x2_tmp_sq = torch.squeeze(x2_tmp, 2)  # remove temporal dimension

        x3 = self.single_conv_block1_1x1(x2_tmp_sq)
        # return input of TAM (x2_down) & ASPP block (x3) & Latent space (x3)
        return x2_down, x3


class TARSSNet_V1(nn.Module):
    """ 
    Temporal Multi-View with ASPP Network (TMVA-Net)

    PARAMETERS
    ----------
    n_classes: int
        Number of classes used for the semantic segmentation task
    n_frames: int
        Total numer of frames used as a sequence
    """

    def __init__(self, n_classes, n_frames):
        super().__init__()
        self.n_classes = n_classes
        self.n_frames = n_frames

        # Backbone (encoding)
        self.rd_encoding_branch = EncodingBranch('range_doppler', self.n_frames)
        self.ra_encoding_branch = EncodingBranch('range_angle', self.n_frames)
        self.ad_encoding_branch = EncodingBranch('angle_doppler', self.n_frames)

        # Temporal-Aware Module
        self.rd_tam_branch = TAM(in_ch=128, out_ch=128, signal_type='range_doppler')
        self.ra_tam_branch = TAM(in_ch=128, out_ch=128, signal_type='range_angle')
        self.ad_tam_branch = TAM(in_ch=128, out_ch=128, signal_type='angle_doppler')

        # ASPP Blocks
        self.rd_aspp_block = ASPPBlock(in_ch=128, out_ch=128)
        self.ra_aspp_block = ASPPBlock(in_ch=128, out_ch=128)
        self.ad_aspp_block = ASPPBlock(in_ch=128, out_ch=128)
        self.rd_single_conv_block1_1x1 = ConvBlock(in_ch=640, out_ch=128, k_size=1, pad=0, dil=1)
        self.ra_single_conv_block1_1x1 = ConvBlock(in_ch=640, out_ch=128, k_size=1, pad=0, dil=1)
        self.ad_single_conv_block1_1x1 = ConvBlock(in_ch=640, out_ch=128, k_size=1, pad=0, dil=1)

        # TAM ASPP Fusion
        self.rd_single_conv_block1_1x1_tsf = ConvBlock(in_ch=256, out_ch=128, k_size=1, pad=0, dil=1)
        self.ra_single_conv_block1_1x1_tsf = ConvBlock(in_ch=256, out_ch=128, k_size=1, pad=0, dil=1)
        self.ad_single_conv_block1_1x1_tsf = ConvBlock(in_ch=256, out_ch=128, k_size=1, pad=0, dil=1)

        # Decoding
        self.rd_single_conv_block2_1x1 = ConvBlock(in_ch=384, out_ch=128, k_size=1, pad=0, dil=1)
        self.ra_single_conv_block2_1x1 = ConvBlock(in_ch=384, out_ch=128, k_size=1, pad=0, dil=1)

        # Pallel range-Doppler (RD) and range-angle (RA) decoding branches
        self.rd_upconv1 = nn.ConvTranspose2d(384, 128, (2, 1), stride=(2, 1))
        self.ra_upconv1 = nn.ConvTranspose2d(384, 128, 2, stride=2)
        self.rd_double_conv_block1 = DoubleConvBlock(in_ch=128, out_ch=128, k_size=3,
                                                     pad=1, dil=1)
        self.ra_double_conv_block1 = DoubleConvBlock(in_ch=128, out_ch=128, k_size=3,
                                                     pad=1, dil=1)
        self.rd_upconv2 = nn.ConvTranspose2d(128, 128, (2, 1), stride=(2, 1))
        self.ra_upconv2 = nn.ConvTranspose2d(128, 128, 2, stride=2)
        self.rd_double_conv_block2 = DoubleConvBlock(in_ch=128, out_ch=128, k_size=3,
                                                     pad=1, dil=1)
        self.ra_double_conv_block2 = DoubleConvBlock(in_ch=128, out_ch=128, k_size=3,
                                                     pad=1, dil=1)

        # Final 1D convs
        self.rd_final = nn.Conv2d(in_channels=128, out_channels=n_classes, kernel_size=1)
        self.ra_final = nn.Conv2d(in_channels=128, out_channels=n_classes, kernel_size=1)


    def forward(self, x_rd, x_ra, x_ad):
        # Backbone
        rd_features, rd_latent = self.rd_encoding_branch(x_rd)
        ra_features, ra_latent = self.ra_encoding_branch(x_ra)
        ad_features, ad_latent = self.ad_encoding_branch(x_ad)

        # TAM
        rd_tmp_aug = self.rd_tam_branch(rd_features)
        ra_tmp_aug = self.ra_tam_branch(ra_features)
        ad_tmp_aug = self.ad_tam_branch(ad_features)

        # ASPP blocks
        x1_rd = self.rd_aspp_block(rd_latent)
        x1_ra = self.ra_aspp_block(ra_latent)
        x1_ad = self.ad_aspp_block(ad_latent)
        x2_rd = self.rd_single_conv_block1_1x1(x1_rd)
        x2_ra = self.ra_single_conv_block1_1x1(x1_ra)
        x2_ad = self.ad_single_conv_block1_1x1(x1_ad)

        # Temporal-Spatial Fusion
        rd_tam_aspp = torch.cat((rd_tmp_aug, x2_rd), 1)
        ra_tam_aspp = torch.cat((ra_tmp_aug, x2_ra), 1)
        ad_tam_aspp = torch.cat((ad_tmp_aug, x2_ad), 1)
        rd_tsf = self.rd_single_conv_block1_1x1_tsf(rd_tam_aspp)
        ra_tsf = self.ra_single_conv_block1_1x1_tsf(ra_tam_aspp)
        ad_tsf = self.ad_single_conv_block1_1x1_tsf(ad_tam_aspp)

        # Latent Space
        # Features join either the RD or the RA branch
        x3 = torch.cat((rd_latent, ra_latent, ad_latent), 1)
        x3_rd = self.rd_single_conv_block2_1x1(x3)
        x3_ra = self.ra_single_conv_block2_1x1(x3)

        # Latent Space + TSF features
        x4_rd = torch.cat((rd_tsf, x3_rd, ad_tsf), 1)
        x4_ra = torch.cat((ra_tsf, x3_ra, ad_tsf), 1)

        # Parallel decoding branches with upconvs
        x5_rd = self.rd_upconv1(x4_rd)
        x5_ra = self.ra_upconv1(x4_ra)
        x6_rd = self.rd_double_conv_block1(x5_rd)
        x6_ra = self.ra_double_conv_block1(x5_ra)

        x7_rd = self.rd_upconv2(x6_rd)
        x7_ra = self.ra_upconv2(x6_ra)
        x8_rd = self.rd_double_conv_block2(x7_rd)
        x8_ra = self.ra_double_conv_block2(x7_ra)

        # Final 1D convolutions
        x9_rd = self.rd_final(x8_rd)
        x9_ra = self.ra_final(x8_ra)

        return x9_rd, x9_ra
