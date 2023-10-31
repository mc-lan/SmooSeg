import torch
from utils import *
import torch.nn.functional as F
import models.dino.vision_transformer as vits

class DinoFeaturizer(nn.Module):

    def __init__(self, dim, cfg):
        super().__init__()
        self.cfg = cfg
        self.dim = dim
        patch_size = self.cfg.dino_patch_size
        self.patch_size = patch_size
        self.feat_type = self.cfg.dino_feat_type
        arch = self.cfg.model_type
        self.model = vits.__dict__[arch](patch_size=patch_size, num_classes=0)
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

        if arch == "vit_small" and patch_size == 16:
            url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
        elif arch == "vit_small" and patch_size == 8:
            url = "dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"
        elif arch == "vit_base" and patch_size == 16:
            url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
        elif arch == "vit_base" and patch_size == 8:
            url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
        else:
            raise ValueError("Unknown arch and patch size")

        print("Since no pretrained weights have been provided, we load the reference pretrained DINO weights.")
        state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
        self.model.load_state_dict(state_dict, strict=True)


    def forward(self, img, n=1, return_class_feat=False):
        self.model.eval()
        with torch.no_grad():
            assert (img.shape[2] % self.patch_size == 0)
            assert (img.shape[3] % self.patch_size == 0)

            # get selected layer activations
            feat, attn, qkv = self.model.get_intermediate_feat(img, n=n)
            feat, attn, qkv = feat[0], attn[0], qkv[0]

            feat_h = img.shape[2] // self.patch_size
            feat_w = img.shape[3] // self.patch_size

            if self.feat_type == "feat":
                image_feat = feat[:, 1:, :].reshape(feat.shape[0], feat_h, feat_w, -1).permute(0, 3, 1, 2).contiguous()
            elif self.feat_type == "KK":
                image_k = qkv[1, :, :, 1:, :].reshape(feat.shape[0], 6, feat_h, feat_w, -1)
                B, H, I, J, D = image_k.shape
                image_feat = image_k.permute(0, 1, 4, 2, 3).reshape(B, H * D, I, J)
                # image_feat = qkv[:, 1:, :].reshape(feat.shape[0], feat_h, feat_w, -1).permute(0, 3, 1, 2)
            else:
                raise ValueError("Unknown feat type:{}".format(self.feat_type))

            if return_class_feat:
                return feat[:, :1, :].reshape(feat.shape[0], 1, 1, -1).permute(0, 3, 1, 2)

        return image_feat


class DinoV2Featurizer(nn.Module):

    def __init__(self, dim, cfg):
        super().__init__()
        self.cfg = cfg
        self.dim = dim
        self.patch_size = self.cfg.dino_patch_size
        arch = self.cfg.model_type

        if arch == "vit_small":
            self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        elif arch == "vit_base":
            self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        elif arch == "vit_large":
            self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
        elif arch == "vit_giant":
            self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14')


        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

    def forward(self, img, n=1):
        self.model.eval()
        with torch.no_grad():
            # get selected layer activations
            feat = self.model.get_intermediate_layers(img, n=n, reshape=True)
            image_feat = feat[0]

        return image_feat

class Projection(nn.Module):

    def __init__(self, cfg):
        super(Projection, self).__init__()
        self.cfg = cfg
        self.dim = cfg.dim
        self.n_feats = cfg.n_feats
        self.dropout = torch.nn.Dropout2d(p=0.2)

        self.cluster1 = self.make_clusterer(self.n_feats)
        self.proj_type = cfg.projection_type
        if self.proj_type == "nonlinear":
            self.cluster2 = self.make_nonlinear_clusterer(self.n_feats)

    def make_clusterer(self, in_channels):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, self.dim, (1, 1))
        )

    def make_nonlinear_clusterer(self, in_channels):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, in_channels, (1, 1)),
            torch.nn.SiLU(),
            torch.nn.Conv2d(in_channels, self.dim, (1, 1)))


    def forward(self, feat):
        if self.proj_type is not None:
            code = self.cluster1(self.dropout(feat))
            if self.proj_type == "nonlinear":
                code += self.cluster2(self.dropout(feat))
        else:
            code = feat

        if self.cfg.dropout:
            return self.dropout(feat), code
        else:
            return feat, code

class Prediction(nn.Module):

    def __init__(self, cfg, n_classes: int, sigma=False):
        super(Prediction, self).__init__()
        self.n_classes = n_classes
        self.dim = cfg.dim
        # self.local_clusters = nn.init.kaiming_normal_(torch.nn.Parameter(torch.randn(self.n_classes, self.dim)))
        # self.local_clusters = nn.init.orthogonal_(torch.nn.Parameter(torch.randn(self.n_classes, self.dim)))
        self.local_clusters = nn.init.xavier_normal_(torch.nn.Parameter(torch.randn(self.n_classes, self.dim)))
        self.init_global_clusters()
        self.alpha = cfg.alpha
        if sigma:
            self.sigma = nn.Parameter(torch.Tensor(1))
            self.sigma.data.fill_(1)
        else:
            self.register_parameter('sigma', None)

    def init_global_clusters(self):
        self.local_clusters.data = F.normalize(self.local_clusters, dim=1)
        self.global_clusters = torch.nn.Parameter(self.local_clusters.data.clone())
        self.global_clusters.requires_grad = False

    def update_global_clusters(self):
        self.global_clusters.data = self.alpha * self.global_clusters.data + (1 - self.alpha) * self.local_clusters.data

    def reset_parameters(self):
        with torch.no_grad():
            self.local_clusters = nn.init.xavier_normal_(torch.nn.Parameter(torch.randn(self.n_classes, self.dim)))

    def forward(self, x):
        normed_local_clusters = F.normalize(self.local_clusters, dim=1)
        normed_global_clusters = F.normalize(self.global_clusters, dim=1)
        normed_features = F.normalize(x, dim=1)

        inner_products_local = torch.einsum("bchw,nc->bnhw", normed_features.detach(), normed_local_clusters)
        inner_products_global = torch.einsum("bchw,nc->bnhw", normed_features, normed_global_clusters.detach())

        if self.sigma is not None:
            inner_products_local = self.sigma * inner_products_local

        return inner_products_local, inner_products_global

class FeaturePyramidNet(nn.Module):
    def __init__(self, cut_model):
        super(FeaturePyramidNet, self).__init__()
        self.layer_nums = [5, 6, 7]
        self.spatial_resolutions = [7, 14, 28, 56]
        self.feat_channels = [2048, 1024, 512, 3]
        for p in cut_model.parameters():
            p.requires_grad = False
        self.encoder = NetWithActivations(cut_model, self.layer_nums)

    def forward(self, x):
        with torch.no_grad():
            feats = self.encoder(x)
        low_res_feats = feats[self.layer_nums[-1]]

        return low_res_feats

class NetWithActivations(torch.nn.Module):
    def __init__(self, model, layer_nums):
        super(NetWithActivations, self).__init__()
        self.layers = nn.ModuleList(model.children())
        self.layer_nums = []
        for l in layer_nums:
            if l < 0:
                self.layer_nums.append(len(self.layers) + l)
            else:
                self.layer_nums.append(l)
        self.layer_nums = set(sorted(self.layer_nums))

    def forward(self, x):
        activations = {}
        for ln, l in enumerate(self.layers):
            x = l(x)
            if ln in self.layer_nums:
                activations[ln] = x
        return activations


def norm(t):
    return F.normalize(t, dim=1, eps=1e-10)


def tensor_correlation(a, b):
    return torch.einsum("nchw,ncij->nhwij", a, b)


def sample(t: torch.Tensor, coords: torch.Tensor):
    return F.grid_sample(t, coords.permute(0, 2, 1, 3), padding_mode='border', align_corners=True)


@torch.jit.script
def super_perm(size: int, device: torch.device):
    perm = torch.randperm(size, device=device, dtype=torch.long)
    perm[perm == torch.arange(size, device=device)] += 1
    return perm % size


class Energy_minimization_loss(nn.Module):
    def __init__(self, cfg, n_classes):
        super(Energy_minimization_loss, self).__init__()
        self.cfg = cfg
        self.n_classes = n_classes
        self.smooth_loss = ContrastiveCorrelationLoss(cfg)

    def forward(self, signal, inner_products_local, inner_products_global, temperature=0.1):
        cluster_probs = torch.softmax(inner_products_global / temperature, dim=1)
        pos_intra_loss, pos_intra_cd, neg_inter_loss, neg_inter_cd = self.smooth_loss(signal, cluster_probs)
        smooth_loss = pos_intra_loss + neg_inter_loss

        # cluster_probs_global = F.one_hot(torch.argmax(inner_products_global, dim=1), self.n_classes) \
        #     .permute(0, 3, 1, 2).to(torch.float32)
        # data_loss = -(cluster_probs_global * inner_products_local).sum(1).mean()

        target = torch.argmax(inner_products_global, dim=1)

        flat_logits = inner_products_local.permute(0, 2, 3, 1).reshape(-1, self.n_classes)
        flat_target = target.reshape(-1)

        data_loss = F.cross_entropy(flat_logits, flat_target, reduction='none')

        return smooth_loss, data_loss.mean(), pos_intra_cd, neg_inter_cd

class ContrastiveCorrelationLoss(nn.Module):
    def __init__(self, cfg, ):
        super(ContrastiveCorrelationLoss, self).__init__()
        self.cfg = cfg

    def helper(self, f1, f2, c1, c2, shift):
        with torch.no_grad():
            fd = tensor_correlation(norm(f1), norm(f2))

            if self.cfg.pointwise:
                old_mean = fd.mean()
                fd -= fd.mean([3, 4], keepdim=True)
                fd = fd - fd.mean() + old_mean

        cd = 1 - tensor_correlation(norm(c1), norm(c2))
        loss = (fd - shift) * cd

        return loss, cd

    def forward(self, orig_feats: torch.Tensor, orig_code: torch.Tensor):
        perm_neg = super_perm(orig_feats.size(0), orig_feats.device)
        feats_neg = orig_feats[perm_neg]
        code_neg = orig_code[perm_neg]

        pos_intra_loss, pos_intra_cd = self.helper(orig_feats, orig_feats, orig_code, orig_code, self.cfg.pos_intra_shift)
        neg_inter_loss, neg_inter_cd = self.helper(orig_feats, feats_neg, orig_code, code_neg, self.cfg.neg_inter_shift)

        return pos_intra_loss.mean(), pos_intra_cd.mean(), neg_inter_loss.mean(), neg_inter_cd.mean()



