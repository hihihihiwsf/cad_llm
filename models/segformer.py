from transformers import SegformerForSemanticSegmentation, SegformerFeatureExtractor
import pytorch_lightning as pl
from datasets import load_metric
import torch
from PIL import Image


class SegformerModel(pl.LightningModule):

    def __init__(self):
        super().__init__()

        # self.save_hyperparameters()

        self.num_classes = 2
        self.metrics_interval = 500

        self.model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b0-finetuned-ade-512-512",
            return_dict=False,
            num_labels=self.num_classes,
            ignore_mismatched_sizes=True,
        )

        self.train_mean_iou = load_metric("mean_iou")
        self.val_mean_iou = load_metric("mean_iou")

        self.val_output_list = []

        self.lr = 3e-4

        self.weighted_loss = torch.nn.CrossEntropyLoss(weight=torch.tensor([0.5, 256.]))

    def forward(self, pixel_values, labels):
        unused_loss, logits = self.model(pixel_values=pixel_values, labels=labels)

        # Flatten
        batch_size, num_classes, res_x, res_y = logits.shape
        flat_logits = logits.view(batch_size, num_classes, res_x * res_y)
        flat_labels = labels.view(batch_size, res_x * res_y)

        weighted_loss = self.weighted_loss(input=flat_logits, target=flat_labels)

        return weighted_loss, logits

    def training_step(self, batch, batch_idx):
        weighted_loss, logits = self.forward(pixel_values=batch["pixel_values"], labels=batch["labels"])

        stats = {"loss": weighted_loss}

        predicted = logits.argmax(dim=1)

        self.train_mean_iou.add_batch(
            predictions=predicted.detach().cpu().numpy(),
            references=batch["labels"].detach().cpu().numpy(),
        )

        if batch_idx % self.metrics_interval == 0:
            more_stats = self.train_mean_iou.compute(
                num_labels=self.num_classes,
                ignore_index=255,
                reduce_labels=False,
            )

            # keys: 'mean_iou', 'mean_accuracy', 'overall_accuracy', 'per_category_iou', 'per_category_accuracy'
            stats["iou_background"] = more_stats["per_category_iou"][0]
            stats["iou_vertices"] = more_stats["per_category_iou"][1]

            stats["acc_background"] = more_stats["per_category_accuracy"][0]
            stats["acc_vertices"] = more_stats["per_category_accuracy"][1]

        self.log_stats(stats)

        return stats

    def validation_step(self, batch, batch_idx):
        weighted_loss, logits = self.forward(pixel_values=batch["pixel_values"], labels=batch["labels"])

        predicted = logits.argmax(dim=1)

        self.val_mean_iou.add_batch(
            predictions=predicted.detach().cpu().numpy(),
            references=batch["labels"].detach().cpu().numpy()
        )

        pred_vert_count = predicted.sum(axis=(1, 2))

        self.val_output_list.append({"val_loss": weighted_loss, "val_pred_vert_count": pred_vert_count})

        # Log images to comet if comet is enabled
        if batch_idx < 4 and len(self.loggers) == 3:
            np_image = torch.cat([batch["labels"], predicted], dim=-1).type(torch.uint8).numpy()
            np_image = (1 - np_image) * 255

            batch_size = predicted.shape[0]
            for i in range(batch_size):
                self.loggers[2].experiment.log_image(
                    image_data=Image.fromarray(np_image[i], mode="L"),
                    name=f"epoch_{self.current_epoch}_sample_{batch_idx * batch_size + i}.png"
                )

    def on_validation_epoch_end(self):
        stats = self.get_stats(self.val_mean_iou)
        stats = {"val_" + k: v for k, v in stats.items()}

        avg_val_loss = torch.stack([x["val_loss"] for x in self.val_output_list]).mean()
        stats["val_loss"] = avg_val_loss

        avg_val_pred_vert_count = torch.stack([x["val_pred_vert_count"] for x in self.val_output_list]).mean(dtype=float)
        stats["val_pred_vert_count"] = avg_val_pred_vert_count

        self.log_stats(stats)
        self.val_output_list = []

        return stats

    def get_stats(self, metric):
        stats = metric.compute(
            num_labels=self.num_classes,
            ignore_index=255,
            reduce_labels=False,
        )

        return {
            "mean_iou": stats["mean_iou"],
            "mean_accuracy": stats["mean_accuracy"],
            "iou_background": stats["per_category_iou"][0],
            "iou_vertices": stats["per_category_iou"][1],
            "acc_background": stats["per_category_accuracy"][0],
            "acc_vertices": stats["per_category_accuracy"][1],
        }

    def log_stats(self, stats, on_step=False, on_epoch=True):
        for k, v in stats.items():
            self.log(k, v, on_step=on_step, on_epoch=on_epoch, prog_bar=False, logger=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad], lr=self.lr)
        return optimizer
