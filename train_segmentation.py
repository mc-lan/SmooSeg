from utils import *
from modules import *
from data import *
from torch.utils.data import DataLoader
import torch.nn.functional as F
from datetime import datetime
import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from lightning_fabric.utilities.seed import seed_everything
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint
import sys
import torch.multiprocessing as mp
from multiprocessing import Pool

torch.multiprocessing.set_sharing_strategy('file_system')
class LitUnsupervisedSegmenter(pl.LightningModule):
    def __init__(self, n_classes, cfg):
        super().__init__()
        self.cfg = cfg
        self.n_classes = n_classes

        data_dir = join(cfg.output_root, "data")
        if cfg.arch == "feature-pyramid":
            cut_model = load_model(cfg.model_type, data_dir)
            self.net = FeaturePyramidNet(cut_model)
        elif cfg.arch == "dino":
            self.net = DinoFeaturizer(cfg.dim, cfg)
        elif cfg.arch == "dinov2":
            self.net = DinoV2Featurizer(cfg.dim, cfg)
        else:
            raise ValueError("Unknown arch {}".format(cfg.arch))

        self.projection = Projection(cfg)
        self.prediction = Prediction(cfg, n_classes + cfg.extra_clusters)

        self.cluster_metrics = UnsupervisedMetrics(
            "test/cluster/", n_classes, cfg.extra_clusters, True)

        self.test_cluster_metrics = UnsupervisedMetrics(
            "final/cluster/", n_classes, cfg.extra_clusters, True)

        self.energy_minimization_loss = Energy_minimization_loss(cfg, n_classes + cfg.extra_clusters)

        self.automatic_optimization = True

        if self.cfg.dataset_name.startswith("cityscapes"):
            self.label_cmap = create_cityscapes_colormap()
        elif self.cfg.dataset_name.startswith("cocostuff27"):
            self.label_cmap = create_pascal_label_colormap()
        else:
            self.label_cmap = create_potsdam_colormap()

        self.val_steps = 0
        self.save_hyperparameters()

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        return self.net(x)[1]

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward

        with torch.no_grad():
            img = batch["img"]
            feats1 = self.net(img)

        log_args = dict(sync_dist=True, rank_zero_only=True)

        feats, code = self.projection(feats1)

        inner_products_local, inner_products_global = self.prediction(code)

        smooth_loss, data_loss, pos_intra_cd, neg_inter_cd = self.energy_minimization_loss(feats,
                                                               inner_products_local,
                                                               inner_products_global,
                                                               temperature=self.cfg.temperature
                                                               )
        loss = smooth_loss + data_loss

        # should_log_hist = (self.cfg.hist_freq is not None) and \
        #                   (self.global_step % self.cfg.hist_freq == 0) and \
        #                   (self.global_step > 0)
        # if should_log_hist:
        #     self.logger.experiment.add_histogram("intra_cd", pos_intra_cd, self.global_step)
        #     self.logger.experiment.add_histogram("neg_cd", neg_inter_cd, self.global_step)

        self.log('loss/data_loss', data_loss, **log_args)
        self.log('loss/smooth_loss', smooth_loss, **log_args)
        self.log('loss/total', loss, **log_args)
        self.log('loss/lr1', self.optimizers().optimizer.param_groups[0]['lr'], **log_args)
        self.log('loss/lr2', self.optimizers().optimizer.param_groups[1]['lr'], **log_args)

        self.prediction.update_global_clusters()

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam([{'params': list(self.projection.parameters()), 'lr': self.cfg.lr1},
                                      {'params': list(self.prediction.parameters()), 'lr': self.cfg.lr2}])
        return optimizer

    def on_train_start(self):
        tb_metrics = {
            **self.cluster_metrics.compute()
        }
        self.logger.log_hyperparams(self.cfg, tb_metrics)

    def validation_step(self, batch, batch_idx):
        img = batch["img"]
        label = batch["label"]
        self.net.eval()
        with torch.no_grad():
            feats = self.net(img)
            _, code = self.projection(feats)

            code = F.interpolate(code, label.shape[-2:], mode='bilinear', align_corners=False)

            _, products = self.prediction(code)

            cluster_probs = torch.log_softmax(products * 2, dim=1)
            cluster_preds = cluster_probs.argmax(1)

            self.cluster_metrics.update(cluster_preds, label)

            return {
                'img': img[:self.cfg.n_images].detach().cpu(),
                "cluster_preds": cluster_preds[:self.cfg.n_images].detach().cpu(),
                "label": label[:self.cfg.n_images].detach().cpu()}

    def on_validation_epoch_end(self) -> None:
        super().on_validation_epoch_end()
        with torch.no_grad():
            tb_metrics = {
                **self.cluster_metrics.compute(),
            }
            self.log_dict(tb_metrics, sync_dist=True)

            if self.trainer.is_global_zero and self.cfg.azureml_logging:
                from azureml.core.run import Run
                run_logger = Run.get_context()
                for metric, value in tb_metrics.items():
                    run_logger.log(metric, value)

            self.cluster_metrics.reset()

    def on_test_start(self) -> None:
        self.test_cluster_metrics.reset()

    def test_step(self, batch, batch_idx):
        img = batch["img"]
        label = batch["label"]
        self.net.eval()
        with Pool(self.cfg.num_workers) as pool:
            with torch.no_grad():
                feats = self.net(img)
                _, code = self.projection(feats)
                code = F.interpolate(code, label.shape[-2:], mode='bilinear', align_corners=False)
                _, products = self.prediction(code)
                cluster_probs = torch.log_softmax(products * 2, dim=1)
                cluster_preds = batched_crf(pool, img, cluster_probs).argmax(1).cuda()

                self.test_cluster_metrics.update(cluster_preds, label)

    def on_test_epoch_end(self) -> None:
        super().on_test_epoch_end()

        with torch.no_grad():
            tb_metrics = {
                **self.test_cluster_metrics.compute(),
            }
            for k, v in tb_metrics.items():
                print(k, ': ', v)
            self.log_dict(tb_metrics, sync_dist=True)


@hydra.main(config_path="configs", config_name="train_config.yaml", version_base='1.1')
def my_app(cfg: DictConfig) -> None:
    OmegaConf.set_struct(cfg, False)
    print(OmegaConf.to_yaml(cfg))
    data_dir = cfg.data_dir
    log_dir = join(cfg.output_root, "logs")
    checkpoint_dir = join(cfg.output_root, "checkpoints")

    prefix = "{}/{}_{}".format(cfg.log_dir, cfg.dataset_name, cfg.experiment_name)
    name = '{}_date_{}'.format(prefix, datetime.now().strftime('%b%d_%H-%M-%S'))
    cfg.full_name = cfg.output_root

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    seed_everything(seed=cfg.seed)

    sys.stdout.flush()

    train_dataset = ContrastiveSegDataset(
        data_dir=data_dir,
        dataset_name=cfg.dataset_name,
        crop_type=cfg.crop_type,
        image_set="train",
        transform=get_transform(cfg.res, False, cfg.loader_crop_type),
        target_transform=get_transform(cfg.res, True, cfg.loader_crop_type),
        cfg=cfg,
        mask=True
    )

    if cfg.dataset_name == "voc":
        val_loader_crop = None
    else:
        val_loader_crop = "center"

    val_dataset = ContrastiveSegDataset(
        data_dir=data_dir,
        dataset_name=cfg.dataset_name,
        crop_type=None,
        image_set="val",
        transform=get_transform(cfg.res, False, val_loader_crop),
        target_transform=get_transform(cfg.res, True, val_loader_crop),
        mask=True,
        cfg=cfg,
    )

    train_loader = DataLoader(train_dataset, cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
                              persistent_workers = True, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, cfg.batch_size, shuffle=False, num_workers=cfg.num_workers,
                            persistent_workers = True, pin_memory=True, drop_last=True)

    if cfg.dataset_name == "potsdam":
        res = 224
    else:
        res = 320

    loader_crop = "center"
    test_dataset = ContrastiveSegDataset(
        data_dir=data_dir,
        dataset_name=cfg.dataset_name,
        crop_type=None,
        image_set="val",
        transform=get_transform(res, False, loader_crop),
        target_transform=get_transform(res, True, loader_crop),
        mask=True,
        cfg=cfg,
    )

    test_loader = DataLoader(test_dataset, cfg.batch_size, shuffle=False, num_workers=cfg.num_workers,
                             persistent_workers = True, pin_memory=True)

    model = LitUnsupervisedSegmenter(train_dataset.n_classes, cfg)

    tb_logger = TensorBoardLogger(
        join(log_dir, name),
        default_hp_metric=False
    )

    gpu_args = dict(accelerator='gpu', devices=-1, val_check_interval=cfg.val_freq)

    if gpu_args["val_check_interval"] > len(train_loader):
        gpu_args.pop("val_check_interval")

    trainer = Trainer(
        enable_progress_bar=True,
        log_every_n_steps=cfg.scalar_log_freq,
        logger=tb_logger,
        max_steps=cfg.max_steps,
        strategy=DDPStrategy(find_unused_parameters=True),
        precision=16,
        callbacks=[
            ModelCheckpoint(
                dirpath=join(checkpoint_dir, name),
                every_n_train_steps=cfg.checkpoint_freq,
                save_top_k=1,
                monitor="test/cluster/mIoU",
                mode="max",
            )
        ],
        **gpu_args
    )
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader, ckpt_path="best")


if __name__ == "__main__":
    mp.set_start_method('spawn')
    prep_args()
    my_app()
