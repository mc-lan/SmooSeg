from modules import *
from data import *
from collections import defaultdict
from multiprocessing import Pool
import hydra
import torch.multiprocessing
from crf import dense_crf
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.multiprocessing as mp
from train_segmentation import LitUnsupervisedSegmenter

torch.multiprocessing.set_sharing_strategy('file_system')

def _apply_crf(tup):
    return dense_crf(tup[0], tup[1])


def batched_crf(pool, img_tensor, prob_tensor):
    outputs = pool.map(_apply_crf, zip(img_tensor.detach().cpu(), prob_tensor.detach().cpu()))
    return torch.cat([torch.from_numpy(arr).unsqueeze(0) for arr in outputs], dim=0)


@hydra.main(config_path="configs", config_name="eval_config.yaml", version_base='1.1')
def my_app(cfg: DictConfig) -> None:
    data_dir = cfg.data_dir
    result_dir = "../results/predictions/{}".format(cfg.experiment_name)
    os.makedirs(join(result_dir, "img"), exist_ok=True)
    os.makedirs(join(result_dir, "label"), exist_ok=True)
    os.makedirs(join(result_dir, "cluster"), exist_ok=True)

    for model_path in cfg.model_paths:
        model = LitUnsupervisedSegmenter.load_from_checkpoint(model_path)

        loader_crop = "center"
        test_dataset = ContrastiveSegDataset(
            data_dir=data_dir,
            dataset_name=model.cfg.dataset_name,
            crop_type=None,
            image_set="val",
            transform=get_transform(cfg.res, False, loader_crop),
            target_transform=get_transform(cfg.res, True, loader_crop),
            cfg=model.cfg,
            mask=True,
        )

        test_loader = DataLoader(test_dataset, cfg.batch_size,
                                 shuffle=False, num_workers=cfg.num_workers,
                                 pin_memory=True)

        model.eval().cuda()

        if cfg.use_ddp:
            par_model = torch.nn.DataParallel(model.net)
            par_projection = torch.nn.DataParallel(model.projection)
            par_prediction = torch.nn.DataParallel(model.prediction)
        else:
            par_model = model.net
            par_projection = model.projection
            par_prediction = model.prediction

        if model.cfg.dataset_name == "cocostuff27":
            all_good_images = range(2500)
        elif model.cfg.dataset_name == "cityscapes":
            all_good_images = range(600)
        elif  model.cfg.dataset_name == "potsdam":
            all_good_images = range(900)
        else:
            raise ValueError("Unknown Dataset {}".format(model.cfg.dataset_name))
        batch_nums = torch.tensor([n // (cfg.batch_size) for n in all_good_images])
        batch_offsets = torch.tensor([n % (cfg.batch_size) for n in all_good_images])

        saved_data = defaultdict(list)
        with Pool(cfg.num_workers + 5) as pool:
            for i, batch in enumerate(tqdm(test_loader)):
                with torch.no_grad():

                    img = batch["img"].cuda()
                    label = batch["label"].cuda()
                    image_index = batch['mask']

                    feats1 = par_model(img)
                    feats2 = par_model(img.flip(dims=[3]))
                    _, code1 = par_projection(feats1)
                    _, code2 = par_projection(feats2)

                    code = (code1 + code2.flip(dims=[3])) / 2

                    code = F.interpolate(code, label.shape[-2:], mode='bilinear', align_corners=False)
                    _, products = par_prediction(code)
                    cluster_probs = torch.log_softmax(products * 2, dim=1)

                    if cfg.run_crf:
                        cluster_preds = batched_crf(pool, img, cluster_probs).argmax(1).cuda()
                    else:
                        cluster_preds = cluster_probs.argmax(1)


                    model.test_cluster_metrics.update(cluster_preds, label)

                    # if i in batch_nums:
                    #     matching_offsets = batch_offsets[torch.where(batch_nums == i)]
                    #     for offset in matching_offsets:
                    #         saved_data["cluster_preds"].append(cluster_preds.cpu()[offset].unsqueeze(0))
                    #         saved_data["label"].append(label.cpu()[offset].unsqueeze(0))
                    #         saved_data["img"].append(img.cpu()[offset].unsqueeze(0))
                    #         saved_data["name"].append(image_index[0])

        tb_metrics = {
            **model.test_cluster_metrics.compute(),
        }

        # for ii in range(len(saved_data["cluster_preds"])):
        #     plot_img = (prep_for_plot(saved_data["img"][ii].cpu().squeeze(0)) * 255).numpy().astype(np.uint8)
        #     plot_label = (model.label_cmap[saved_data["label"][ii].cpu().squeeze(0)]).astype(np.uint8)
        #     Image.fromarray(plot_img).save(join(join(result_dir, "img", saved_data["name"][ii] + ".jpg")))
        #     Image.fromarray(plot_label).save(join(join(result_dir, "label", saved_data["name"][ii] + ".png")))
        #     plot_cluster = (model.label_cmap[
        #         model.test_cluster_metrics.map_clusters(saved_data["cluster_preds"][ii].cpu().squeeze(0))]).astype(np.uint8)
        #     Image.fromarray(plot_cluster).save(join(join(result_dir, "cluster", saved_data["name"][ii] + ".png")))

        print(model_path)
        print(tb_metrics)

if __name__ == "__main__":
    mp.set_start_method('spawn')
    prep_args()
    my_app()