from .detector3d_template import Detector3DTemplate


class PointRCNN(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    def forward(self, batch_dict):
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {"loss": loss}
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}
        loss_point, tb_dict = self.point_head.get_loss()
        loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)

        if (
            hasattr(self, "history_query")
            and self.history_query is not None
            and self.history_query.model_cfg.get("P2_LOSS_CONFIG", None) is not None
        ):
            loss_p2, tb_dict = self.history_query.get_p2_loss(tb_dict)
            disp_dict.update(
                {
                    "det_loss": (loss_point.item() + loss_rcnn.item()),
                }
            )
            if self.history_query.model_cfg.P2_LOSS_CONFIG.P2_ONLY:
                loss = loss_p2
            else:
                loss = loss_point + loss_rcnn + loss_p2
        else:
            loss = loss_point + loss_rcnn

        return loss, tb_dict, disp_dict
