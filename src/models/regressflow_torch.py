import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tm

# With small adjustments taken from 
# @misc{li2021humanposeregressionresidual,
#       title={Human Pose Regression with Residual Log-likelihood Estimation}, 
#       author={Jiefeng Li and Siyuan Bian and Ailing Zeng and Can Wang and Bo Pang and Wentao Liu and Cewu Lu},
#       year={2021},
#       eprint={2107.11291},
#       archivePrefix={arXiv},
#       primaryClass={cs.CV},
#       url={https://arxiv.org/abs/2107.11291}, 
# }

class Linear(nn.Module):
    def __init__(self, in_channel, out_channel, bias=True, norm=True):
        super(Linear, self).__init__()
        self.bias = bias
        self.norm = norm
        self.linear = nn.Linear(in_channel, out_channel, bias)
        nn.init.xavier_uniform_(self.linear.weight, gain=0.01)

    def forward(self, x):
        y = x.matmul(self.linear.weight.t())
        if self.norm:
            x_norm = torch.norm(x, dim=1, keepdim=True)
            y = y / x_norm
        if self.bias:
            y = y + self.linear.bias
        return y

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=nn.BatchNorm2d):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(planes, momentum=0.1)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = norm_layer(planes, momentum=0.1)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * 4, momentum=0.1)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, architecture, norm_layer=nn.BatchNorm2d):
        super(ResNet, self).__init__()
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.block = Bottleneck
        self.layers = [3, 4, 6, 3]  # for resnet50

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(64, eps=1e-5, momentum=0.1, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self.make_layer(self.block, 64, self.layers[0])
        self.layer2 = self.make_layer(self.block, 128, self.layers[1], stride=2)
        self.layer3 = self.make_layer(self.block, 256, self.layers[2], stride=2)
        self.layer4 = self.make_layer(self.block, 512, self.layers[3], stride=2)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                self._norm_layer(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, norm_layer=self._norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=self._norm_layer))
        return nn.Sequential(*layers)

class RegressFlow(nn.Module):
    def __init__(self, PRESET, NUM_LAYERS, NUM_FC_FILTERS, HIDDEN_LIST, PRETRAINED, TRY_LOAD):
        super(RegressFlow, self).__init__()
        self._preset_cfg = PRESET
        self.fc_dim = NUM_FC_FILTERS
        self.num_joints = self._preset_cfg['NUM_JOINTS']
        self.height_dim = self._preset_cfg['IMAGE_SIZE'][0]
        self.width_dim = self._preset_cfg['IMAGE_SIZE'][1]

        self.preact = ResNet(f"resnet{NUM_LAYERS}")

        # Imagenet pretrain model
        x = tm.resnet50(pretrained=True)
        self.feature_channel = 2048

        model_state = self.preact.state_dict()
        state = {k: v for k, v in x.state_dict().items()
                 if k in self.preact.state_dict() and v.size() == self.preact.state_dict()[k].size()}
        model_state.update(state)
        self.preact.load_state_dict(model_state)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fcs, out_channel = self._make_fc_layer()
        self.fc_coord = Linear(out_channel, self.num_joints * 2)
        self.fc_sigma = Linear(out_channel, self.num_joints * 2, norm=False)
        self.fc_sigma2 = Linear(out_channel, self.num_joints, norm=False)
        self.fc_layers = [self.fc_coord, self.fc_sigma]

    def _make_fc_layer(self):
        fc_layers = []
        num_deconv = len(self.fc_dim)
        input_channel = self.feature_channel
        for i in range(num_deconv):
            if self.fc_dim[i] > 0:
                fc = nn.Linear(input_channel, self.fc_dim[i])
                bn = nn.BatchNorm1d(self.fc_dim[i])
                fc_layers.append(fc)
                fc_layers.append(bn)
                fc_layers.append(nn.ReLU(inplace=True))
                input_channel = self.fc_dim[i]
            else:
                fc_layers.append(nn.Identity())
        return nn.Sequential(*fc_layers), input_channel

    def forward(self, x):
        BATCH_SIZE = x.shape[0]
        feat = self.preact(x)
        feat = self.avg_pool(feat).reshape(BATCH_SIZE, -1)
        out_coord = self.fc_coord(feat).reshape(BATCH_SIZE, self.num_joints, 2)

        
        out_log_variance = self.fc_sigma(feat).reshape(BATCH_SIZE, self.num_joints, -1)
        raw_cov_xy = self.fc_sigma2(feat).reshape(BATCH_SIZE, self.num_joints)
        pred_jts = out_coord.reshape(BATCH_SIZE, self.num_joints, 2)
        log_variance = out_log_variance
        log_var_x = out_log_variance[:, :, 0]  # (B, N)
        log_var_y = out_log_variance[:, :, 1]  # (B, N)

        # Compute variances
        var_x = torch.exp(log_var_x)  # Ensure positivity
        var_y = torch.exp(log_var_y)

        # Constrain cov_xy to maintain positive-definiteness
        cov_xy = torch.tanh(raw_cov_xy) * torch.sqrt(var_x * var_y)  # (B, N)
        sigma = torch.exp(0.5 * log_variance)
        pure_sigma = log_variance
        scores = 1 - torch.sigmoid(log_variance)
        scores = torch.mean(scores, dim=2, keepdim=True)
        return {
            'feat': feat, # debug
            'pred_jts': pred_jts,
            'sigma': sigma,
            'log_variance': log_variance, # we need
            'covariance': cov_xy,    # we need
            'maxvals': scores.float(),
            'nf_loss': None,
            'pure_sigma': pure_sigma
        }

def build_model(cfg):
    model = RegressFlow(
        PRESET=cfg['PRESET'],
        NUM_LAYERS=cfg['NUM_LAYERS'],
        NUM_FC_FILTERS=cfg['NUM_FC_FILTERS'],
        HIDDEN_LIST=cfg['HIDDEN_LIST'],
        PRETRAINED=cfg['PRETRAINED'],
        TRY_LOAD=cfg['TRY_LOAD']
    )
    return model
