import torch
import torch.nn as nn


def conv_bn(inp, oup, kernel, stride, padding=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel, stride, padding, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True),
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, use_res_connect, expand_ratio=6):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.use_res_connect = use_res_connect

        self.conv = nn.Sequential(
            nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                inp * expand_ratio,
                inp * expand_ratio,
                3,
                stride,
                1,
                groups=inp * expand_ratio,
                bias=False,
            ),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU(inplace=True),
            nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class PFLDInference(nn.Module):
    def __init__(self, channel_scale: float = 1.0):
        super().__init__()

        """
        Channel Scale check
        채널 scale이 0보다 작은 것을 검토합니다.
        """
        if channel_scale < 0.0:
            raise Exception("Channel Scale must be larger then 0.")

        """
        FPLD 내에서 MobileNet Block은 2개의 in/out 채널을 가지며
        in/out에 따라 동적으로 내부 inverted의 채널이 결정되기 때문에
        Block 외부 주입만으로 충분합니다.
        model_channel_list는 기존에 정의되어있던 두 상수를 의미합니다.
        """
        model_channel_list = [64, 128]
        channel_scale_denominator = int(1 / channel_scale)
        """
        아래의 코드는 model scale이 적용 가능한 지 검토하며,
        적용하지 못할 경우 에러를 출력합니다.
        """
        for num_model_channel in model_channel_list:
            if num_model_channel % channel_scale_denominator != 0:
                print(f"num_model_channel: {num_model_channel}")
                print(f"channel_scale_denominator: {channel_scale_denominator}")
                print(f"remain: {num_model_channel % channel_scale_denominator}")
                raise Exception("channel scale can not applied.")

        scaled_channel_list = [
            x // channel_scale_denominator for x in model_channel_list
        ]

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        """
        in default(channel_scale=1, 64 and 128 in inverted residual(NobileNet blocks))
        """
        self.conv3_1 = InvertedResidual(64, scaled_channel_list[0], 2, False, 2)

        self.block3_2 = InvertedResidual(
            scaled_channel_list[0], scaled_channel_list[0], 1, True, 2
        )
        self.block3_3 = InvertedResidual(
            scaled_channel_list[0], scaled_channel_list[0], 1, True, 2
        )
        self.block3_4 = InvertedResidual(
            scaled_channel_list[0], scaled_channel_list[0], 1, True, 2
        )
        self.block3_5 = InvertedResidual(
            scaled_channel_list[0], scaled_channel_list[0], 1, True, 2
        )

        """Stride 2 Block(Downsampling)"""
        self.conv4_1 = InvertedResidual(
            scaled_channel_list[0], scaled_channel_list[1], 2, False, 2
        )

        self.conv5_1 = InvertedResidual(
            scaled_channel_list[1], scaled_channel_list[1], 1, False, 4
        )
        self.block5_2 = InvertedResidual(
            scaled_channel_list[1], scaled_channel_list[1], 1, True, 4
        )
        self.block5_3 = InvertedResidual(
            scaled_channel_list[1], scaled_channel_list[1], 1, True, 4
        )
        self.block5_4 = InvertedResidual(
            scaled_channel_list[1], scaled_channel_list[1], 1, True, 4
        )
        self.block5_5 = InvertedResidual(
            scaled_channel_list[1], scaled_channel_list[1], 1, True, 4
        )
        self.block5_6 = InvertedResidual(
            scaled_channel_list[1], scaled_channel_list[1], 1, True, 4
        )

        self.conv6_1 = InvertedResidual(
            scaled_channel_list[1], 16, 1, False, 2
        )  # [16, 14, 14]

        self.conv7 = conv_bn(16, 32, 3, 2)  # [32, 7, 7]
        self.conv8 = nn.Conv2d(32, 128, 7, 1, 0)  # [128, 1, 1]
        self.bn8 = nn.BatchNorm2d(128)

        self.avg_pool1 = nn.AvgPool2d(14)
        self.avg_pool2 = nn.AvgPool2d(7)
        self.fc = nn.Linear(176, 196)

    def forward(self, x):  # x: 3, 112, 112
        x = self.relu(self.bn1(self.conv1(x)))  # [64, 56, 56]
        x = self.relu(self.bn2(self.conv2(x)))  # [64, 56, 56]
        x = self.conv3_1(x)
        x = self.block3_2(x)
        x = self.block3_3(x)
        x = self.block3_4(x)
        out1 = self.block3_5(x)

        x = self.conv4_1(out1)
        x = self.conv5_1(x)
        x = self.block5_2(x)
        x = self.block5_3(x)
        x = self.block5_4(x)
        x = self.block5_5(x)
        x = self.block5_6(x)
        x = self.conv6_1(x)
        x1 = self.avg_pool1(x)
        x1 = x1.view(x1.size(0), -1)

        x = self.conv7(x)
        x2 = self.avg_pool2(x)
        x2 = x2.view(x2.size(0), -1)

        x3 = self.relu(self.conv8(x))
        x3 = x3.view(x3.size(0), -1)

        multi_scale = torch.cat([x1, x2, x3], 1)
        landmarks = self.fc(multi_scale)

        # return out1, landmarks
        return landmarks


class AuxiliaryNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = conv_bn(64, 128, 3, 2)
        self.conv2 = conv_bn(128, 128, 3, 1)
        self.conv3 = conv_bn(128, 32, 3, 2)
        self.conv4 = conv_bn(32, 128, 7, 1)
        self.max_pool1 = nn.MaxPool2d(3)
        self.fc1 = nn.Linear(128, 32)
        self.fc2 = nn.Linear(32, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.max_pool1(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)

        return x
