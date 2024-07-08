import numpy as np
import torch
import torch.nn as nn


class BaseBEVBackbone(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg

        if self.model_cfg.get("LAYER_NUMS", None) is not None:
            assert (
                len(self.model_cfg.LAYER_NUMS)
                == len(self.model_cfg.LAYER_STRIDES)
                == len(self.model_cfg.NUM_FILTERS)
            )
            layer_nums = self.model_cfg.LAYER_NUMS
            layer_strides = self.model_cfg.LAYER_STRIDES
            num_filters = self.model_cfg.NUM_FILTERS
        else:
            layer_nums = layer_strides = num_filters = []

        if self.model_cfg.get("UPSAMPLE_STRIDES", None) is not None:
            assert len(self.model_cfg.UPSAMPLE_STRIDES) == len(
                self.model_cfg.NUM_UPSAMPLE_FILTERS
            )
            num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
            upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
        else:
            upsample_strides = num_upsample_filters = []

        if (self.model_cfg.get("P2_LOSS_CONFIG", None)) is not None:
            self.p2_backbone = nn.Sequential(
                nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1)
            )

        num_levels = len(layer_nums)
        c_in_list = [input_channels, *num_filters[:-1]]
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        for idx in range(num_levels):
            cur_layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(
                    c_in_list[idx],
                    num_filters[idx],
                    kernel_size=3,
                    stride=layer_strides[idx],
                    padding=0,
                    bias=False,
                ),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ]
            for k in range(layer_nums[idx]):
                cur_layers.extend(
                    [
                        nn.Conv2d(
                            num_filters[idx],
                            num_filters[idx],
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                        nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU(),
                    ]
                )
            self.blocks.append(nn.Sequential(*cur_layers))
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride >= 1:
                    self.deblocks.append(
                        nn.Sequential(
                            nn.ConvTranspose2d(
                                num_filters[idx],
                                num_upsample_filters[idx],
                                upsample_strides[idx],
                                stride=upsample_strides[idx],
                                bias=False,
                            ),
                            nn.BatchNorm2d(
                                num_upsample_filters[idx], eps=1e-3, momentum=0.01
                            ),
                            nn.ReLU(),
                        )
                    )
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_filters[idx],
                                num_upsample_filters[idx],
                                stride,
                                stride=stride,
                                bias=False,
                            ),
                            nn.BatchNorm2d(
                                num_upsample_filters[idx], eps=1e-3, momentum=0.01
                            ),
                            nn.ReLU(),
                        )
                    )

        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > num_levels:
            self.deblocks.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        c_in,
                        c_in,
                        upsample_strides[-1],
                        stride=upsample_strides[-1],
                        bias=False,
                    ),
                    nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                    nn.ReLU(),
                )
            )

        self.num_bev_features = c_in

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        spatial_features = data_dict["spatial_features"]
        ups = []
        ret_dict = {}
        x = spatial_features
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)

            stride = int(spatial_features.shape[2] / x.shape[2])
            ret_dict["spatial_features_%dx" % stride] = x
            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x))
            else:
                ups.append(x)

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]

        if len(self.deblocks) > len(self.blocks):
            x = self.deblocks[-1](x)

        data_dict["spatial_features_2d"] = x

        if self.model_cfg.get("P2_LOSS_CONFIG", None) is not None:
            # print("computed p2 loss")
            _, _, p2_features = self.break_up_pc(data_dict["points_voxels"])
            self.p2_scores = torch.tensor(data_dict["p2_score"])
            self.p2_pred = self.p2_backbone(p2_features)

        return data_dict

    def break_up_pc(self, pc):
        batch_idx = pc[:, 0]
        xyz = pc[:, 1:4].contiguous()
        features = pc[:, 4:].contiguous() if pc.size(-1) > 4 else None
        return batch_idx, xyz, features

    def get_loss(self):
        p2_loss = self.get_p2_loss(self.p2_pred.squeeze(), self.p2_scores)

        return p2_loss

    def get_p2_loss(self, p2_pred, p2_gt):
        assert self.model_cfg.get("P2_LOSS_CONFIG", None) is not None
        p2_loss_cfgs = self.model_cfg.P2_LOSS_CONFIG

        if p2_loss_cfgs.LOSS_FN == "ce":
            loss = nn.CrossEntropyLoss()
        elif p2_loss_cfgs.LOSS_FN == "nll":
            loss = nn.NLLLoss()
        elif p2_loss_cfgs.LOSS_FN == "mse":
            loss = nn.MSELoss()
        elif p2_loss_cfgs.LOSS_FN == "l1":
            loss = nn.L1Loss()
        else:
            raise NotImplementedError

        p2_loss = loss(p2_pred, p2_gt)
        return p2_loss
