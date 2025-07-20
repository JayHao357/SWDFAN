import torch.nn as nn
import torch.nn.functional as F
import torch


class FEModule(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates):
        super(FEModule, self).__init__()
        self.convb = nn.Conv2d(in_channels[0], out_channels, kernel_size=1)
        self.atrous_conv1 = nn.Conv2d(in_channels[1], out_channels, dilation=atrous_rates[0], kernel_size=3,
                                      padding=atrous_rates[0])
        self.atrous_conv2 = nn.Conv2d(in_channels[2], out_channels, dilation=atrous_rates[1], kernel_size=3,
                                      padding=atrous_rates[1])

        # self.con_bn_relu = ConvBNReLU(in_channels=24, out_channels=8)

    def forward(self, x):
        res1, res2, res3 = x
        f1 = self.convb(res1)
        f1 = F.avg_pool2d(f1, kernel_size=4, stride=4)

        f2 = self.atrous_conv1(res2)
        f2 = F.avg_pool2d(f2, kernel_size=2, stride=2)

        f3 = self.atrous_conv2(res3)
        f3 = F.avg_pool2d(f3, kernel_size=1, stride=1)

        fem_out = torch.cat([f1, f2, f3], dim=1)
        # final_out = self.con_bn_relu(fem_out)
        return fem_out


class PPMBilinear(nn.Module):
    def __init__(self, num_classes=8, fc_dim=2048,
                 use_aux=False, pool_scales=(1, 2, 3, 6),
                 norm_layer=nn.BatchNorm2d
                 ):
        super(PPMBilinear, self).__init__()
        self.use_aux = use_aux
        self.ppm = []
        # self.layer6 = FEModule(in_channels=[256, 512, 1024], out_channels=8, atrous_rates=[6, 12])

        for scale in pool_scales:
            self.ppm.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),
                norm_layer(512),
                nn.ReLU(inplace=True)
            ))
        self.ppm = nn.ModuleList(self.ppm)
        if self.use_aux:
            self.cbr_deepsup = nn.Sequential(
                nn.Conv2d(fc_dim // 2, fc_dim // 4, kernel_size=3, stride=1,
                          padding=1, bias=False),
                norm_layer(fc_dim // 4),
                nn.ReLU(inplace=True),
            )
            self.conv_last_deepsup = nn.Conv2d(fc_dim // 4, num_classes, 1, 1, 0)
            self.dropout_deepsup = nn.Dropout2d(0.1)

        self.conv_last = nn.Sequential(
            nn.Conv2d(fc_dim + len(pool_scales) * 512, 512,
                      kernel_size=3, padding=1, bias=False),
            norm_layer(512),
            nn.ReLU(inplace=True),
            # nn.Dropout2d(0.1),
            nn.Dropout2d(0.5),
            nn.Conv2d(512, num_classes, kernel_size=1)
        )

    def forward(self, conv_out):
        # conv5 = conv_out[-1]
        res1, res2, res3, res4 = conv_out
        # fem_in = [res1, res2, res3]
        conv_out = res4
        # fem_out = self.layer6(fem_in)
        input_size = conv_out.size()
        ppm_out = [conv_out]
        for pool_scale in self.ppm:
            ppm_out.append(F.interpolate(
                pool_scale(conv_out),
                (input_size[2], input_size[3]),
                mode='bilinear', align_corners=False))
        # ppm_out.append(fem_out)
        ppm_out = torch.cat(ppm_out, 1)

        x = self.conv_last(ppm_out)
        if self.use_aux and self.training:
            conv4 = conv_out[-2]
            _ = self.cbr_deepsup(conv4)
            _ = self.dropout_deepsup(_)
            _ = self.conv_last_deepsup(_)

            return x
        else:
            return x


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d,
                 bias=False):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels),
            nn.ReLU6()
        )


class Classifier_Module(nn.Module):
    def __init__(self, inplanes, dilation_series, padding_series, num_classes):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(
                nn.Conv2d(inplanes, num_classes, kernel_size=3, stride=1, padding=padding, dilation=dilation,
                          bias=True))

        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list) - 1):
            out += self.conv2d_list[i + 1](x)
        return out


class SpeASpaModule(nn.Module):
    def __init__(self, kernelsize=3, n=64, dim=3, size=[13, 13]):
        super(SpeASpaModule, self).__init__()
        stride = (kernelsize - 1) // 2
        self.conv_spa1 = nn.Conv2d(dim, 3, 1, 1)
        self.conv_spa2 = nn.Conv2d(3, n, 1, 1)
        self.conv_spe1 = nn.Conv2d(dim, n, size[0], 1)
        self.conv_spe2 = nn.ConvTranspose2d(n, n, size[0])
        self.conv1 = nn.Conv2d(n + n, n, kernelsize, 1, stride)
        self.conv2 = nn.Conv2d(n, dim, kernelsize, 1, stride)

    def forward(self, x):
        x_spa = F.relu(self.conv_spa1(x))
        x_spe = F.relu(self.conv_spe1(x))
        x_spe = self.conv_spe2(x_spe)
        x_spa = self.conv_spa2(x_spa)

        x = F.relu(self.conv1(torch.cat((x_spa, x_spe), 1)))
        x = torch.sigmoid(self.conv2(x))

        return x


from module.resnet import ResNetEncoder
import ever as er


class Deeplabv2(er.ERModule):
    def __init__(self, config):
        super(Deeplabv2, self).__init__(config)
        self.encoder = ResNetEncoder(self.config.backbone)
        self.spa_e = SpeASpaModule(n=64, kernelsize=3, dim=3, size=[13, 13])
        self.conv1 = nn.Conv2d(6, 3, kernel_size=1)
        if self.config.multi_layer:
            print('Use multi_layer!')
            if self.config.cascade:
                if self.config.use_ppm:
                    self.layer5 = PPMBilinear(**self.config.ppm1)
                    self.layer6 = PPMBilinear(**self.config.ppm2)
                else:
                    self.layer5 = self._make_pred_layer(Classifier_Module, self.config.inchannels // 2, [6, 12, 18, 24],
                                                        [6, 12, 18, 24], self.config.num_classes)
                    self.layer6 = self._make_pred_layer(Classifier_Module, self.config.inchannels, [6, 12, 18, 24],
                                                        [6, 12, 18, 24], self.config.num_classes)
            else:
                if self.config.use_ppm:
                    self.layer5 = PPMBilinear(**self.config.ppm)
                    self.layer6 = PPMBilinear(**self.config.ppm)
                    # self.layer5 = FEModule(in_channels=[256, 512, 1024], out_channels=7, atrous_rates=[6, 12])
                    # self.layer6 = FEModule(in_channels=[256, 512, 1024], out_channels=7, atrous_rates=[6, 12])
                else:
                    self.layer5 = self._make_pred_layer(Classifier_Module, self.config.inchannels, [6, 12, 18, 24],
                                                        [6, 12, 18, 24], self.config.num_classes)
                    self.layer6 = self._make_pred_layer(Classifier_Module, self.config.inchannels, [6, 12, 18, 24],
                                                        [6, 12, 18, 24], self.config.num_classes)
        else:
            if self.config.use_ppm:
                self.cls_pred = PPMBilinear(**self.config.ppm)
            else:
                self.cls_pred = self._make_pred_layer(Classifier_Module, self.config.inchannels, [6, 12, 18, 24],
                                                      [6, 12, 18, 24], self.config.num_classes)

    def _make_pred_layer(self, block, inplanes, dilation_series, padding_series, num_classes):
        return block(inplanes, dilation_series, padding_series, num_classes)

    def forward(self, x):
        B, C, H, W = x.shape
        if self.config.multi_layer:
            if self.config.cascade:
                feat1, feat2 = self.encoder(x)[-2:]
                x1 = self.layer5(feat1)
                x2 = self.layer6(feat2)
                if self.training:
                    return x1, feat1, x2, feat2
                else:
                    x1 = F.interpolate(x1, (H, W), mode='bilinear', align_corners=True)
                    x2 = F.interpolate(x2, (H, W), mode='bilinear', align_corners=True)
                    return (x1.softmax(dim=1) + x2.softmax(dim=1)) / 2
            else:
                # feat = self.encoder(x)[-1]
                e1, e2, e3, feat = self.encoder(x)
                fem_in = [e1, e2, e3, feat]

                x1 = self.layer5(fem_in)
                x2 = self.layer6(fem_in)
                if self.training:
                    return x1, x2, feat
                else:
                    x1 = F.interpolate(x1, (H, W), mode='bilinear', align_corners=True)
                    x2 = F.interpolate(x2, (H, W), mode='bilinear', align_corners=True)
                    return (x1.softmax(dim=1) + x2.softmax(dim=1)) / 2

        else:
            feat = self.encoder(x)[-1]
            x = self.cls_pred(feat)
            if self.training:
                return x, feat
            else:
                x = F.interpolate(x, (H, W), mode='bilinear', align_corners=True)
                return x.softmax(dim=1)

    def set_default_config(self):
        self.config.update(dict(
            backbone=dict(
                resnet_type='resnet50',
                output_stride=16,
                pretrained=True,
            ),
            multi_layer=False,
            cascade=False,
            use_ppm=False,
            ppm=dict(
                num_classes=8,
                use_aux=False,
                norm_layer=nn.BatchNorm2d,

            ),
            inchannels=2048,
            num_classes=8
        ))


if __name__ == '__main__':
    model = Deeplabv2(dict(
        backbone=dict(
            resnet_type='resnet50',
            output_stride=16,
            pretrained=True,
        ),
        multi_layer=False,
        cascade=False,
        use_ppm=True,
        ppm=dict(
            num_classes=8,
            use_aux=False,
            fc_dim=2048,
        ),
        inchannels=2048,
        num_classes=8
    ))
    input = torch.rand(2, 3, 512, 512)
    model.train()
    outputs = model(input)
    print(outputs[0].shape)
    print(outputs[1].shape)

    label = torch.randint(low=0, high=7, size=(2, 512, 512)).long()
    print(label.max())
    label = F.one_hot(label, num_classes=8)
    label = label.permute(0, 3, 1, 2)
    print(label.shape)