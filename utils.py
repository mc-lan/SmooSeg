import collections
import os
from os.path import join
import io

import numpy as np
import torch.multiprocessing
import torch.nn as nn
import torch.nn.functional as F
import wget
from PIL import Image
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import np_str_obj_array_pattern, default_collate_err_msg_format
from torchmetrics import Metric
from torchvision import models
from torchvision import transforms as T
from torch.utils.tensorboard.summary import hparams
from einops import rearrange
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from crf import dense_crf
import cv2
import random
import torchvision

def convert_from_color(arr_3d):
    """ RGB-color encoding to grayscale labels """

    potsdam = {0: (255, 255, 255),  # roads (white)
               1: (0, 0, 255),  # clutter (blue)
               2: (0, 255, 255),  # car (cyan)
               3: (0, 255, 0),  # Tree (green)
               4: (255, 255, 0),  # Low vegetation (yellow)
               5: (255, 0, 0)}  # Buildings (red)

    invert_palette = {v: k for k, v in potsdam.items()}

    arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)

    for c, i in invert_palette.items():
        m = np.all(arr_3d == np.array(c).reshape(1, 1, 3), axis=2)
        arr_2d[m] = i

    return arr_2d

def generate_test_sample():
    imagedir = '/data/mclan/datasets/potsdam_original'
    patch_size = 320
    patch_number = 13
    image_transform = T.Compose([T.CenterCrop(size=patch_size*patch_number),
                                 T.Resize(size=224*patch_number),
                                 T.ToTensor(),
                                 T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                 ])
    label_transform = T.Compose([T.RandomCrop(size=patch_size*patch_number),
                                 T.Resize(size=224 * patch_number),
                                 ToTargetTensor()
                                 ])

    fine_to_coarse = {0: 0, 2: 0,  # roads and cars
                      1: 1, 5: 1,  # buildings and clutter
                      3: 2, 4: 2,  # vegetation and trees
                      }

    image = Image.open(os.path.join(imagedir, '2_Ortho_RGB', 'top_potsdam_3_10_RGB.tif'))  # 6000--> 5152 = 224 * 23
    label = cv2.imread(os.path.join(imagedir, '5_Labels_all', 'top_potsdam_3_10_label.tif'))
    label =Image.fromarray(label)

    # left = 2200
    # top = 300
    left = 0
    top = 0

    right= left + patch_size * patch_number
    bottom = top + patch_size * patch_number
    image = image.crop((left, top, right, bottom))
    label = label.crop((left, top, right, bottom))

    label = convert_from_color(np.array(label))

    new_label_map = np.zeros_like(label)
    for fine, coarse in fine_to_coarse.items():
        new_label_map[label == fine] = coarse
    label = new_label_map

    label = Image.fromarray(label)

    seed = np.random.randint(2147483647)
    random.seed(seed)
    torch.manual_seed(seed)
    image = image_transform(image)

    random.seed(seed)
    torch.manual_seed(seed)
    label = label_transform(label).squeeze(0)

    image = rearrange(image, 'c (h1 h) (w1 w) -> (h1 w1) c h w', h=224, w=224)
    label = rearrange(label, '(h1 h) (w1 w) -> (h1 w1) h w', h=224, w=224)

    return image, label


def potsdam_distribution_plot(img, dino, smooseg, stego, label, preds, center_dino, center_stego, center_smooseg,j):

    imgs = []
    for i in range(img.shape[0]):
        imgs.append(unnorm(img[i]) * 255)
    imgs = torch.stack(imgs)
    img = torchvision.utils.make_grid(imgs, nrow=5)
    img = Image.fromarray(np.uint8(img.permute(1, 2, 0).detach().cpu().numpy()), 'RGB')

    label = rearrange(label, 'b (h p1) (w p2) -> b h w (p1 p2)', p1=8, p2=8)
    label = torch.mode(label, dim=3).values.flatten().detach().cpu().numpy()

    # preds = rearrange(preds, 'b (h p1) (w p2) -> b h w (p1 p2)', p1=8, p2=8)
    # preds = torch.mode(preds, dim=3).values.flatten().detach().cpu().numpy()
    #
    # preds = np.array([assignment[1][preds[i]] for i in range(len(preds))])

    dino = F.normalize(dino, dim=1)
    smooseg = F.normalize(smooseg, dim=1)
    stego = F.normalize(stego, dim=1)

    dino = rearrange(dino, 'b c h w -> (b h w) c')
    smooseg = rearrange(smooseg, 'b c h w -> (b h w) c')
    stego = rearrange(stego, 'b c h w -> (b h w) c')

    center_dino = F.normalize(center_dino, dim=1)
    center_stego = F.normalize(center_stego, dim=1)
    center_smooseg = F.normalize(center_smooseg, dim=1)

    dino = torch.cat([center_dino, dino]).detach().cpu().numpy()
    stego = torch.cat([center_stego, stego]).detach().cpu().numpy()
    smooseg = torch.cat([center_smooseg, smooseg]).detach().cpu().numpy()

    dino_embedded = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3, metric='cosine').fit_transform(dino)
    smooseg_embedded = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3, metric='cosine').fit_transform(smooseg)
    stego_embedded = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3, metric='cosine').fit_transform(stego)

    center_dino_embedded = dino_embedded[0:3]
    center_stego_embedded = stego_embedded[0:3]
    center_smooseg_embedded = smooseg_embedded[0:3]

    label_center_dino = np.array([0,2,1])
    label_center_stego = np.array([1,2,0])
    label_center_smooseg = np.array([0,1,2])

    dino_embedded = dino_embedded[3:]
    stego_embedded = stego_embedded[3:]
    smooseg_embedded = smooseg_embedded[3:]


    fig, axs = plt.subplots(1, 4, figsize=(20, 5))

    axs[0].imshow(img)
    axs[0].set_title("Image")

    # set
    index = [2,17,18]
    index_not = list(set(range(25)) - set(index))

    dino_embedded = rearrange(dino_embedded, '(b h w) c -> b c h w', h=28, w=28)
    smooseg_embedded = rearrange(smooseg_embedded, '(b h w) c -> b c h w', h=28, w=28)
    stego_embedded = rearrange(stego_embedded, '(b h w) c -> b c h w', h=28, w=28)
    label = rearrange(label, '(b h w) -> b h w', h=28, w=28)

    label_0 = label[index[0]]
    label_0 = rearrange(label_0, 'h w -> (h w)')
    dino_embedded_0 = dino_embedded[index[0]]
    dino_embedded_0 = rearrange(dino_embedded_0, 'c h w -> (h w) c')
    smooseg_embedded_0 = smooseg_embedded[index[0]]
    smooseg_embedded_0 = rearrange(smooseg_embedded_0, 'c h w -> (h w) c')
    stego_embedded_0 = stego_embedded[index[0]]
    stego_embedded_0 = rearrange(stego_embedded_0, 'c h w -> (h w) c')

    label_1 = label[index[1]]
    label_1 = rearrange(label_1, 'h w -> (h w)')
    dino_embedded_1 = dino_embedded[index[1]]
    dino_embedded_1 = rearrange(dino_embedded_1, 'c h w -> (h w) c')
    smooseg_embedded_1 = smooseg_embedded[index[1]]
    smooseg_embedded_1 = rearrange(smooseg_embedded_1, 'c h w -> (h w) c')
    stego_embedded_1 = stego_embedded[index[1]]
    stego_embedded_1 = rearrange(stego_embedded_1, 'c h w -> (h w) c')

    label_2 = label[index[2]]
    label_2 = rearrange(label_2, 'h w -> (h w)')
    dino_embedded_2 = dino_embedded[index[2]]
    dino_embedded_2 = rearrange(dino_embedded_2, 'c h w -> (h w) c')
    smooseg_embedded_2 = smooseg_embedded[index[2]]
    smooseg_embedded_2 = rearrange(smooseg_embedded_2, 'c h w -> (h w) c')
    stego_embedded_2 = stego_embedded[index[2]]
    stego_embedded_2 = rearrange(stego_embedded_2, 'c h w -> (h w) c')

    label = label[index_not]
    label = rearrange(label, 'b h w -> (b h w)')
    dino_embedded = dino_embedded[index_not]
    dino_embedded = rearrange(dino_embedded, 'b c h w -> (b h w) c')
    smooseg_embedded = smooseg_embedded[index_not]
    smooseg_embedded = rearrange(smooseg_embedded, 'b c h w -> (b h w) c')
    stego_embedded = stego_embedded[index_not]
    stego_embedded = rearrange(stego_embedded, 'b c h w -> (b h w) c')

    axs[1].scatter(dino_embedded[:, 0], dino_embedded[:, 1], s=1, alpha=0.1, c=label)
    axs[1].scatter(dino_embedded_0[:, 0], dino_embedded_0[:, 1], s=15, alpha=0.5, marker='s', c=label_0)
    axs[1].scatter(dino_embedded_1[:, 0], dino_embedded_1[:, 1], s=15, alpha=0.5, marker='*',c=label_1)
    axs[1].scatter(dino_embedded_2[:, 0], dino_embedded_2[:, 1], s=15, alpha=0.5, marker='^',c=label_2)
    axs[1].scatter(center_dino_embedded[:, 0], center_dino_embedded[:, 1], s=[350, 220, 100], alpha=1, marker='p',
                   c=label_center_dino)
    axs[1].set_title("Features of DINO, Acc=69.6%, mIoU=48.6%")

    axs[2].scatter(stego_embedded[:, 0], stego_embedded[:, 1], s=1, alpha=0.1, c=label)
    axs[2].scatter(stego_embedded_0[:, 0], stego_embedded_0[:, 1], s=15, alpha=0.5, marker='s',c=label_0)
    axs[2].scatter(stego_embedded_1[:, 0], stego_embedded_1[:, 1], s=15, alpha=0.5, marker='*',c=label_1)
    axs[2].scatter(stego_embedded_2[:, 0], stego_embedded_2[:, 1], s=15, alpha=0.5, marker='^',c=label_2)
    axs[2].scatter(center_stego_embedded[:, 0], center_stego_embedded[:, 1], s=250, alpha=1, marker='p',
                   c=label_center_stego)
    axs[2].set_title("Embedding of STEGO, Acc=80.0%, mIoU=65.8%")

    axs[3].scatter(smooseg_embedded[:, 0], smooseg_embedded[:, 1], s=1, alpha=0.1, c=label)
    axs[3].scatter(smooseg_embedded_0[:, 0], smooseg_embedded_0[:, 1], s=15, alpha=0.5, marker='s',c=label_0)
    axs[3].scatter(smooseg_embedded_1[:, 0], smooseg_embedded_1[:, 1], s=15, alpha=0.5, marker='*',c=label_1)
    axs[3].scatter(smooseg_embedded_2[:, 0], smooseg_embedded_2[:, 1], s=15, alpha=0.5, marker='^',c=label_2)
    axs[3].scatter(center_smooseg_embedded[:, 0], center_smooseg_embedded[:, 1], s=250, alpha=1, marker='p',
                   c=label_center_smooseg)
    axs[3].set_title("Embedding of SmooSeg, Acc=87.4%, mIoU=77.8%")

    remove_axes(axs)
    plt.tight_layout()
    # plt.show()
    plt.savefig('Visualization_'+str(j)+'.png', dpi=600, bbox_inches='tight')


def get_class_labels(dataset_name):
    if dataset_name.startswith("cityscapes"):
        return [
            'road', 'sidewalk', 'parking', 'rail track', 'building',
            'wall', 'fence', 'guard rail', 'bridge', 'tunnel',
            'pole', 'polegroup', 'traffic light', 'traffic sign', 'vegetation',
            'terrain', 'sky', 'person', 'rider', 'car',
            'truck', 'bus', 'caravan', 'trailer', 'train',
            'motorcycle', 'bicycle']
    elif dataset_name == "cocostuff27":
        return [
            "electronic", "appliance", "food", "furniture", "indoor",
            "kitchen", "accessory", "animal", "outdoor", "person",
            "sports", "vehicle", "ceiling", "floor", "food",
            "furniture", "rawmaterial", "textile", "wall", "window",
            "building", "ground", "plant", "sky", "solid",
            "structural", "water"]
    elif dataset_name == "voc":
        return [
            'background',
            'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
            'bus', 'car', 'cat', 'chair', 'cow',
            'diningtable', 'dog', 'horse', 'motorbike', 'person',
            'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
    elif dataset_name == "potsdam":
        return [
            'roads and cars',
            'buildings and clutter',
            'trees and vegetation']
    else:
        raise ValueError("Unknown Dataset {}".format(dataset_name))

def _apply_crf(tup):
    return dense_crf(tup[0], tup[1])


def batched_crf(pool, img_tensor, prob_tensor):
    outputs = pool.map(_apply_crf, zip(img_tensor.detach().cpu(), prob_tensor.detach().cpu()))
    return torch.cat([torch.from_numpy(arr).unsqueeze(0) for arr in outputs], dim=0)

def prep_for_plot(img, rescale=True, resize=None):
    if resize is not None:
        img = F.interpolate(img.unsqueeze(0), resize, mode="bilinear")
    else:
        img = img.unsqueeze(0)

    plot_img = unnorm(img).squeeze(0).cpu().permute(1, 2, 0)
    if rescale:
        plot_img = (plot_img - plot_img.min()) / (plot_img.max() - plot_img.min())
    return plot_img


def add_plot(writer, name, step):
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg', dpi=100)
    buf.seek(0)
    image = Image.open(buf)
    image = T.ToTensor()(image)
    writer.add_image(name, image, step)
    plt.clf()
    plt.close()


@torch.jit.script
def shuffle(x):
    return x[torch.randperm(x.shape[0])]


def add_hparams_fixed(writer, hparam_dict, metric_dict, global_step):
    exp, ssi, sei = hparams(hparam_dict, metric_dict)
    writer.file_writer.add_summary(exp)
    writer.file_writer.add_summary(ssi)
    writer.file_writer.add_summary(sei)
    for k, v in metric_dict.items():
        writer.add_scalar(k, v, global_step)


@torch.jit.script
def resize(classes: torch.Tensor, size: int):
    return F.interpolate(classes, (size, size), mode="bilinear", align_corners=False)


def one_hot_feats(labels, n_classes):
    return F.one_hot(labels, n_classes).permute(0, 3, 1, 2).to(torch.float32)


def load_model(model_type, data_dir):
    if model_type == "robust_resnet50":
        model = models.resnet50(pretrained=False)
        model_file = join(data_dir, 'imagenet_l2_3_0.pt')
        if not os.path.exists(model_file):
            wget.download("http://6.869.csail.mit.edu/fa19/psets19/pset6/imagenet_l2_3_0.pt",
                          model_file)
        model_weights = torch.load(model_file)
        model_weights_modified = {name.split('model.')[1]: value for name, value in model_weights['model'].items() if
                                  'model' in name}
        model.load_state_dict(model_weights_modified)
        model = nn.Sequential(*list(model.children())[:-1])
    elif model_type == "densecl":
        model = models.resnet50(pretrained=False)
        model_file = join(data_dir, 'densecl_r50_coco_1600ep.pth')
        if not os.path.exists(model_file):
            wget.download("https://cloudstor.aarnet.edu.au/plus/s/3GapXiWuVAzdKwJ/download", model_file)
        model_weights = torch.load(model_file)
        model.load_state_dict(model_weights['state_dict'], strict=False)
        model = nn.Sequential(*list(model.children())[:-1])
    elif model_type == "resnet50":
        model = models.resnet50(pretrained=True)
        model = nn.Sequential(*list(model.children())[:-1])
    elif model_type == "mocov2":
        model = models.resnet50(pretrained=False)
        model_file = join(data_dir, 'moco_v2_800ep_pretrain.pth.tar')
        if not os.path.exists(model_file):
            wget.download("https://dl.fbaipublicfiles.com/moco/moco_checkpoints/"
                          "moco_v2_800ep/moco_v2_800ep_pretrain.pth.tar", model_file)
        checkpoint = torch.load(model_file)
        # rename moco pre-trained keys
        state_dict = checkpoint['state_dict']
        for k in list(state_dict.keys()):
            # retain only encoder_q up to before the embedding layer
            if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                # remove prefix
                state_dict[k[len("module.encoder_q."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]
        msg = model.load_state_dict(state_dict, strict=False)
        assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}
        model = nn.Sequential(*list(model.children())[:-1])
    elif model_type == "densenet121":
        model = models.densenet121(pretrained=True)
        model = nn.Sequential(*list(model.children())[:-1] + [nn.AdaptiveAvgPool2d((1, 1))])
    elif model_type == "vgg11":
        model = models.vgg11(pretrained=True)
        model = nn.Sequential(*list(model.children())[:-1] + [nn.AdaptiveAvgPool2d((1, 1))])
    else:
        raise ValueError("No model: {} found".format(model_type))

    model.eval()
    model.cuda()
    return model

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        image2 = torch.clone(image)
        for t, m, s in zip(image2, self.mean, self.std):
            t.mul_(s).add_(m)
        return image2

normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
unnorm = UnNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


class ToTargetTensor(object):
    def __call__(self, target):
        return torch.as_tensor(np.array(target), dtype=torch.int64).unsqueeze(0)


def prep_args():
    import sys

    old_args = sys.argv
    new_args = [old_args.pop(0)]
    while len(old_args) > 0:
        arg = old_args.pop(0)
        if len(arg.split("=")) == 2:
            new_args.append(arg)
        elif arg.startswith("--"):
            new_args.append(arg[2:] + "=" + old_args.pop(0))
        else:
            raise ValueError("Unexpected arg style {}".format(arg))
    sys.argv = new_args


def get_transform(res, is_label, crop_type):
    if crop_type == "center":
        cropper = T.CenterCrop(res)
    elif crop_type == "random":
        cropper = T.RandomCrop(res)
    elif crop_type is None:
        cropper = T.Lambda(lambda x: x)
        res = (res, res)
    else:
        raise ValueError("Unknown Cropper {}".format(crop_type))
    if is_label:
        return T.Compose([T.Resize(res, Image.NEAREST),
                          cropper,
                          ToTargetTensor()])
    else:
        return T.Compose([T.Resize(res, Image.NEAREST),
                          cropper,
                          T.ToTensor(),
                          normalize])


def _remove_axes(ax):
    ax.xaxis.set_major_formatter(plt.NullFormatter())
    ax.yaxis.set_major_formatter(plt.NullFormatter())
    ax.set_xticks([])
    ax.set_yticks([])


def remove_axes(axes):
    if len(axes.shape) == 2:
        for ax1 in axes:
            for ax in ax1:
                _remove_axes(ax)
    else:
        for ax in axes:
            _remove_axes(ax)


class UnsupervisedMetrics(Metric):
    def __init__(self, prefix: str, n_classes: int, extra_clusters: int, compute_hungarian: bool,
                 dist_sync_on_step=True):
        # call `self.add_state`for every internal state that is needed for the metrics computations
        # dist_reduce_fx indicates the function that should be used to reduce
        # state from multiple processes
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.n_classes = n_classes
        self.extra_clusters = extra_clusters
        self.compute_hungarian = compute_hungarian
        self.prefix = prefix
        self.add_state("stats",
                       default=torch.zeros(n_classes + self.extra_clusters, n_classes, dtype=torch.int64),
                       dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        with torch.no_grad():
            actual = target.reshape(-1)
            preds = preds.reshape(-1)
            mask = (actual >= 0) & (actual < self.n_classes) & (preds >= 0) & (preds < self.n_classes)
            actual = actual[mask]
            preds = preds[mask]
            self.stats += torch.bincount(
                (self.n_classes + self.extra_clusters) * actual + preds,
                minlength=self.n_classes * (self.n_classes + self.extra_clusters)) \
                .reshape(self.n_classes, self.n_classes + self.extra_clusters).t().to(self.stats.device)

    def map_clusters(self, clusters):
        if self.extra_clusters == 0:
            return torch.tensor(self.assignments[1])[clusters]
        else:
            missing = sorted(list(set(range(self.n_classes + self.extra_clusters)) - set(self.assignments[0])))
            cluster_to_class = self.assignments[1]
            for missing_entry in missing:
                if missing_entry == cluster_to_class.shape[0]:
                    cluster_to_class = np.append(cluster_to_class, -1)
                else:
                    cluster_to_class = np.insert(cluster_to_class, missing_entry + 1, -1)
            cluster_to_class = torch.tensor(cluster_to_class)
            return cluster_to_class[clusters]

    def compute(self):
        if self.compute_hungarian:
            self.assignments = linear_sum_assignment(self.stats.detach().cpu(), maximize=True)
            if self.extra_clusters == 0:
                self.histogram = self.stats[np.argsort(self.assignments[1]), :]
            if self.extra_clusters > 0:
                self.assignments_t = linear_sum_assignment(self.stats.detach().cpu().t(), maximize=True)
                histogram = self.stats[self.assignments_t[1], :]
                missing = list(set(range(self.n_classes + self.extra_clusters)) - set(self.assignments[0]))
                new_row = self.stats[missing, :].sum(0, keepdim=True)
                histogram = torch.cat([histogram, new_row], axis=0)
                new_col = torch.zeros(self.n_classes + 1, 1, device=histogram.device)
                self.histogram = torch.cat([histogram, new_col], axis=1)
        else:
            self.assignments = (torch.arange(self.n_classes).unsqueeze(1),
                                torch.arange(self.n_classes).unsqueeze(1))
            self.histogram = self.stats

        tp = torch.diag(self.histogram)
        fp = torch.sum(self.histogram, dim=0) - tp
        fn = torch.sum(self.histogram, dim=1) - tp

        iou = tp / (tp + fp + fn)
        prc = tp / (tp + fn)
        opc = torch.sum(tp) / torch.sum(self.histogram)

        metric_dict = {self.prefix + "mIoU": iou[~torch.isnan(iou)].mean().item(),
                       self.prefix + "Accuracy": opc.item()}

        return {k: 100 * v for k, v in metric_dict.items()}


def flexible_collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage).resize_(0)
        try:
            # out_shape = (len(batch),) + batch[0].shape
            # out.resize_(*out_shape)
            return torch.stack(batch, 0, out=out)
        except RuntimeError:
            return batch
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return flexible_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, str):
    # elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        return {key: flexible_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(flexible_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = zip(*batch)
        return [flexible_collate(samples) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))
