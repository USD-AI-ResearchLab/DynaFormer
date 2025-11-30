import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss
from torchvision import transforms
from utils import test_single_volume
from torch.nn import functional as F
#from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
from calflops import calculate_flops
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import os
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
import imgaug as ia
import imgaug.augmenters as iaa  # 导入iaa

from ptflops import get_model_complexity_info

from torch.optim.lr_scheduler import CyclicLR, StepLR, MultiStepLR, ExponentialLR, CosineAnnealingLR, ReduceLROnPlateau


import torch
import torch.nn as nn
import torch.nn.functional as F
#import pywt

import torch.nn.functional as F
import os, sys, math, logging, random
   
from torchvision import transforms
from tensorboardX import SummaryWriter
    
from utils import  create_edge_mask, plot_result

import torch
import torch.nn as nn
import torch.nn.functional as F

def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovász extension w.r.t sorted errors.
    Args:
        gt_sorted: [P] Tensor, ground truth labels sorted in descending order of errors.
    Returns:
        grad: [P] Tensor, the gradient.
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[:-1]
    return jaccard

def lovasz_softmax_flat(probas, labels, classes='present'):
    """
    Computes Lovász-Softmax loss from flattened predictions and labels.
    Args:
        probas: [P, C] Tensor, class probabilities at each pixel (after softmax).
        labels: [P] Tensor, ground truth labels.
        classes: 'all' or 'present'
    Returns:
        loss: scalar Tensor
    """
    C = probas.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes == 'all' else torch.unique(labels).long().tolist()

    for c in class_to_sum:
        fg = (labels == c).float()  # foreground for class c
        if fg.sum() == 0:
            continue
        probas_c = probas[:, c]
        errors = (fg - probas_c).abs()
        errors_sorted, perm = torch.sort(errors, descending=True)
        fg_sorted = fg[perm]
        grad = lovasz_grad(fg_sorted)
        losses.append(torch.dot(errors_sorted, grad))
    if len(losses) == 0:
        # Only happens if no class is present
        return torch.tensor(0., device=probas.device)
    return torch.mean(torch.stack(losses))

class LovaszSoftmaxLoss(nn.Module):
    def __init__(self, classes='present', per_image=False, ignore_index=None):
        """
        Args:
            classes: 'all' or 'present'
            per_image: compute the loss per image instead of per batch
            ignore_index: label to ignore in loss computation
        """
        super(LovaszSoftmaxLoss, self).__init__()
        self.classes = classes
        self.per_image = per_image
        self.ignore_index = ignore_index

    def forward(self, logits, labels):
        """
        Args:
            logits: [B, C, H, W] Tensor, raw model outputs
            labels: [B, H, W] Tensor, ground truth labels
        Returns:
            loss: scalar Tensor
        """
        if self.per_image:
            loss = torch.mean(torch.stack([
                self.lovasz_softmax_flat(*self.flatten_probas(log.unsqueeze(0), lab.unsqueeze(0)))
                for log, lab in zip(logits, labels)
            ]))
        else:
            loss = self.lovasz_softmax_flat(*self.flatten_probas(logits, labels))
        return loss

    def flatten_probas(self, logits, labels):
        """
        Flattens predictions in the batch and filters out pixels with ignore_index.
        Args:
            logits: [B, C, H, W]
            labels: [B, H, W]
        Returns:
            logits_flat: [P, C] Tensor
            labels_flat: [P] Tensor
        """
        B, C, H, W = logits.size()
        logits = logits.permute(0, 2, 3, 1).contiguous().view(-1, C)
        labels = labels.view(-1)
        if self.ignore_index is not None:
            valid = (labels != self.ignore_index)
            logits = logits[valid]
            labels = labels[valid]
        return logits, labels

    def lovasz_softmax_flat(self, probas, labels):
        return lovasz_softmax_flat(probas, labels, self.classes)

class MultiScalePatchDiscriminator(nn.Module):
    def __init__(self, num_classes=9):
        super(MultiScalePatchDiscriminator, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(num_classes, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 1, 4, padding=1)
        )

    def forward(self, x):
        return self.block(x)

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        
        ce_loss = F.cross_entropy(inputs, targets.long(), reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class HybridSegmentationLoss(nn.Module):
    def __init__(self, num_classes, alpha=0.5, gamma=2.0, edge_weight=1.0):
        """
        Combines Dice, Focal, and edge-aware losses.

        Args:
            num_classes: number of segmentation classes
            alpha: weight for Dice loss (1 - alpha is for Focal loss)
            gamma: focal loss focusing parameter
            edge_weight: additional weight for edge area errors
        """
        super(HybridSegmentationLoss, self).__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma
        self.edge_weight = edge_weight
        self.eps = 1e-6

    def forward(self, logits, targets, edge_mask):
        """
        Args:
            logits: [B, C, H, W] - raw model output
            targets: [B, H, W] - ground truth labels
            edge_mask: [B, 1, H, W] - binary edge map from create_edge_mask()
        Returns:
            hybrid_loss: combined loss value
        """
        probs = torch.softmax(logits, dim=1)
        targets_onehot = F.one_hot(targets.long(), self.num_classes).permute(0, 3, 1, 2).float()

        # ------------------- Dice Loss -------------------
        dims = (0, 2, 3)
        intersection = torch.sum(probs * targets_onehot, dims)
        cardinality = torch.sum(probs + targets_onehot, dims)
        dice_loss = 1 - ((2. * intersection + self.eps) / (cardinality + self.eps)).mean()

        # ------------------- Focal Loss -------------------
        ce_loss = F.cross_entropy(logits, targets.long(), reduction='none')  # [B, H, W]
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()

        # ------------------- Edge-aware Loss -------------------
        edge_mask = edge_mask.squeeze(1)  # [B, H, W]
        edge_loss = (ce_loss * edge_mask).sum() / (edge_mask.sum() + self.eps)

        # Combine all
        hybrid_loss = self.alpha * dice_loss + (1 - self.alpha) * focal_loss + self.edge_weight * edge_loss
        return hybrid_loss

def mask_to_onehot(mask, ):
    """
    Converts a segmentation mask (H, W, C) to (H, W, K) where the last dim is a one
    hot encoding vector, C is usually 1 or 3, and K is the number of class.
    """
    semantic_map = []
    mask = np.expand_dims(mask,-1)
    for colour in range (9):
        equality = np.equal(mask, colour)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1).astype(np.int32)
    return semantic_map

def augment_seg(img_aug, img, seg ):
    seg = mask_to_onehot(seg)
    aug_det = img_aug.to_deterministic() 
    image_aug = aug_det.augment_image( img )

    segmap = ia.SegmentationMapOnImage( seg , nb_classes=np.max(seg)+1 , shape=img.shape )
    segmap_aug = aug_det.augment_segmentation_maps( segmap )
    segmap_aug = segmap_aug.get_arr_int()
    segmap_aug = np.argmax(segmap_aug, axis=-1).astype(np.float32)
    return image_aug , segmap_aug

def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label

def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample


class Synapse_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, img_size, norm_x_transform=None, norm_y_transform=None):
        self.norm_x_transform = norm_x_transform
        self.norm_y_transform = norm_y_transform
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir
        self.img_size = img_size

        self.img_aug = iaa.SomeOf((0,4),[
            iaa.Flipud(0.5, name="Flipud"),
            iaa.Fliplr(0.5, name="Fliplr"),
            iaa.AdditiveGaussianNoise(scale=0.005 * 255),
            iaa.GaussianBlur(sigma=(1.0)),
            iaa.LinearContrast((0.5, 1.5), per_channel=0.5),
            iaa.Affine(scale={"x": (0.5, 2), "y": (0.5, 2)}),
            iaa.Affine(rotate=(-40, 40)),
            iaa.Affine(shear=(-16, 16)),
            iaa.PiecewiseAffine(scale=(0.008, 0.03)),
            iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)})
        ], random_order=True)

        

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, slice_name+'.npz')
            data = np.load(data_path)
            image, label = data['image'], data['label']
            image,label = augment_seg(self.img_aug, image, label)
            x, y = image.shape
            if x != self.img_size or y != self.img_size:
                image = zoom(image, (self.img_size / x, self.img_size / y), order=3)  # why not 3?
                label = zoom(label, (self.img_size / x, self.img_size / y), order=0)

        else:
            vol_name = self.sample_list[idx].strip('\n')
            filepath = self.data_dir + "/{}.npy.h5".format(vol_name)
            data = h5py.File(filepath)
            image, label = data['image'][:], data['label'][:]
        
        

        sample = {'image': image, 'label': label}
        if self.norm_x_transform is not None:
            sample['image'] = self.norm_x_transform(sample['image'].copy())
        if self.norm_y_transform is not None:
            sample['label'] = self.norm_y_transform(sample['label'].copy())
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample
def inference(model, testloader, args, test_save_path=None):
    model.eval()
    metric_list = 0.0

    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        h, w = sampled_batch["image"].size()[2:]
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
        metric_i = test_single_volume(image, label, model, classes=args.num_classes, patch_size=[args.img_size, args.img_size],
                                      test_save_path=test_save_path, case=case_name, z_spacing=args.z_spacing)
        metric_list += np.array(metric_i)
        logging.info(' idx %d case %s mean_dice %f mean_hd95 %f' % (i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))
    
    metric_list = metric_list / len(testloader.dataset)
    
    for i in range(1, args.num_classes):
        logging.info('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i-1][0], metric_list[i-1][1]))
    
    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    
    logging.info('Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))
    
    return performance, mean_hd95


def plot_result(dice, h, snapshot_path,args):
    dict = {'mean_dice': dice, 'mean_hd95': h} 
    df = pd.DataFrame(dict)
    plt.figure(0)
    df['mean_dice'].plot()
    resolution_value = 1200
    plt.title('Mean Dice')
    date_and_time = datetime.datetime.now()
    filename = f'{args.model_name}_' + str(date_and_time)+'dice'+'.png'
    save_mode_path = os.path.join(snapshot_path, filename)
    plt.savefig(save_mode_path, format="png", dpi=resolution_value)
    plt.figure(1)
    df['mean_hd95'].plot()
    plt.title('Mean hd95')
    filename = f'{args.model_name}_' + str(date_and_time)+'hd95'+'.png'
    save_mode_path = os.path.join(snapshot_path, filename)
    #save csv 
    filename = f'{args.model_name}_' + str(date_and_time)+'results'+'.csv'
    save_mode_path = os.path.join(snapshot_path, filename)
    df.to_csv(save_mode_path, sep='\t')
def create_edge_mask(label_batch):
    """
    Generate edge masks from segmentation labels.
    Args:
        label_batch: (Tensor) [B, H, W] containing class indices
    Returns:
        edge_batch: (Tensor) [B, 1, H, W] with 1s at class boundaries
    """
    padded = F.pad(label_batch.float(), (1, 1, 1, 1), mode='replicate')

    left   = padded[:, 1:-1, :-2]
    right  = padded[:, 1:-1, 2:]
    top    = padded[:, :-2, 1:-1]
    bottom = padded[:, 2:, 1:-1]

    # Edge where any neighbor has a different label
    edges = (label_batch != left) | (label_batch != right) | (label_batch != top) | (label_batch != bottom)

    return edges.unsqueeze(1).float()  # [B, 1, H, W]
def create_edge_mask(label_batch):
        if label_batch.dim() == 3:
            label_batch = label_batch.unsqueeze(1)
        label_batch = label_batch.float()
        maxpool = F.max_pool2d(label_batch, kernel_size=3, stride=1, padding=1)
        minpool = -F.max_pool2d(-label_batch, kernel_size=3, stride=1, padding=1)
        edge = maxpool - minpool
        return (edge > 0).float()
    

class HybridSegmentationLoss(nn.Module):
    def __init__(self, num_classes, alpha=0.5, gamma=1.5, edge_weight=1.0, 
                 class_weights=None, smooth=1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha        # Dice vs Focal weight
        self.gamma = gamma        # Focal focusing parameter
        self.edge_weight = edge_weight  # Boundary loss multiplier
        self.smooth = smooth      # Dice smoothing factor
        
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights)
        else:
            self.class_weights = None

    def forward(self, pred, target, edges):
        """
        Args:
            pred: (Tensor) model predictions [B, C, H, W]
            target: (Tensor) ground truth [B, H, W] (class indices)
            edges: (Tensor) boundary mask [B, H, W] (0/1 values)
        Returns:
            (Tensor) combined loss value
        """
        target=target.long()
        # Convert target to one-hot encoding
        target_onehot = F.one_hot(target, self.num_classes).permute(0, 3, 1, 2).float()
        
        # # 1. Multi-class Dice Loss
        
        # d_loss = DiceLoss(self.num_classes)
        # dice_loss=d_loss(pred, label_batch, softmax=True)
        # # Apply class weights if provided
        # if self.class_weights is not None:
        #     dice_loss = dice_loss * self.class_weights.to(pred.device)
        # dice_loss = dice_loss.mean()

        # # 2. Focal Loss (CrossEntropy variant)
        # ce_loss = F.cross_entropy(pred, target, weight=self.class_weights, reduction='none')
        # pt = torch.exp(-ce_loss)
        # focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()

        # log_probs = F.log_softmax(pred, dim=1)
        # ce_boundary = F.nll_loss(log_probs, target,weight=self.class_weights, reduction='none')
        ce_boundary = F.cross_entropy(pred, target, weight=self.class_weights, reduction='mean')  # [B, H, W]
        
        boundary_mask = edges.float()  # [B, H, W]
        boundary_loss = (ce_boundary * boundary_mask).sum() / (boundary_mask.sum())
        boundary_loss = boundary_loss * self.edge_weight
        loss = boundary_loss
        

        # Combine losses
        return loss

def trainer_synapse(args, model, snapshot_path):

    os.makedirs(os.path.join(snapshot_path, 'test'), exist_ok=True)
    test_save_path = os.path.join(snapshot_path, 'test')

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations

    x_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    y_transforms = transforms.ToTensor()

    db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",img_size=args.img_size,
                               norm_x_transform = x_transforms, norm_y_transform = y_transforms)

    print("The length of train set is: {}".format(len(db_train)))
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True,
                             worker_init_fn=worker_init_fn)

    db_test = Synapse_dataset(base_dir=args.test_path, split="test_vol", list_dir=args.list_dir, img_size=args.img_size)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)

    if args.n_gpu > 1:
        model = nn.DataParallel(model)

    model.train()
    # Compute Model Complexity (MACs & Parameters)
    logging.info("\n📌 **Computing Model Complexity:**")
    with torch.cuda.device(0):  # Use GPU if available
        macs, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True, verbose=True)
    
    logging.info(f"📌 **MACs (Multiply-Accumulate Operations):** {macs}")
    logging.info(f"📌 **Trainable Parameters:** {params}")

    print(f"MACs: {macs}")
    print(f"Params: {params}")
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))

    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    dice_=[]
    hd95_= []
    # Optimizer with weight decay
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
   
    focal_loss = FocalLoss(alpha=0.20, gamma=1.0, reduction='mean')
    dice_loss=DiceLoss(num_classes)
    # Weights for combining losses
    focal_loss_weight = 0.3
    dice_loss_weight = 0.7
    ce_loss_weight = 0.3
    
    loss_fn = HybridSegmentationLoss(
    num_classes=9,  # Set your number of classes
    alpha=0.6,      # Dice weight (0-1)
    gamma=2.0,      # Focal parameter
    edge_weight=1.5, # Boundary emphasis
    ).cuda()

    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            # print("data shape---------", image_batch.shape, label_batch.shape)
            image_batch, label_batch = image_batch.cuda(), label_batch.squeeze(1).cuda()
            outputs = model(image_batch)
            edge_batch = create_edge_mask(label_batch)
            outputs = F.interpolate(outputs, size=label_batch.shape[1:], mode='bilinear', align_corners=False)
            # loss_ce = ce_loss(outputs, label_batch[:].long())
            # loss_dice = dice_loss(outputs, label_batch, softmax=True)
            # loss = 0.4 * loss_ce + 0.6 * loss_dice

            loss_focal = focal_loss(outputs, label_batch)
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss =  dice_loss_weight * loss_dice + ce_loss_weight * loss_ce
            #loss = loss_fn(outputs, label_batch, edge_batch)


            # print("loss-----------", loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)

            logging.info('iteration %d : loss : %f, loss_ce: %f, loss_dice: %f' % (iter_num, loss.item(), loss_ce.item(), loss_dice.item()))
      
            if iter_num % 20 == 0:
                image = image_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)
        


        # Test
        eval_interval = args.eval_interval 
        if epoch_num >= int(max_epoch / 2) and (epoch_num + 1) % eval_interval == 0:
            filename = f'{args.model_name}_epoch_{epoch_num}.pth'
            save_mode_path = os.path.join(snapshot_path, filename)
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            
            logging.info("*" * 20)
            logging.info(f"Running Inference after epoch {epoch_num}")
            print(f"Epoch {epoch_num}")
            mean_dice, mean_hd95 = inference(model, testloader, args, test_save_path=test_save_path)
            dice_.append(mean_dice)
            hd95_.append(mean_hd95)
            model.train()

        if epoch_num >= max_epoch - 1:
            filename = f'{args.model_name}_epoch_{epoch_num}.pth'
            save_mode_path = os.path.join(snapshot_path, filename)
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            
            if not (epoch_num + 1) % args.eval_interval == 0:
                logging.info("*" * 20)
                logging.info(f"Running Inference after epoch {epoch_num} (Last Epoch)")
                print(f"Epoch {epoch_num}, Last Epcoh")
                mean_dice, mean_hd95 = inference(model, testloader, args, test_save_path=test_save_path)
                dice_.append(mean_dice)
                hd95_.append(mean_hd95)
                model.train()
                
            iterator.close()
            break
            
    plot_result(dice_, hd95_, snapshot_path, args)
    writer.close()
    return "Training Finished!"
# def trainer_synapse(args, model, snapshot_path):
    

#     os.makedirs(os.path.join(snapshot_path, 'test'), exist_ok=True)
#     test_save_path = os.path.join(snapshot_path, 'test')

#     logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
#                         format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
#     logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
#     logging.info(str(args))

#     base_lr = args.base_lr
#     num_classes = args.num_classes
#     batch_size = args.batch_size * args.n_gpu

#     x_transforms = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize([0.5], [0.5])
#     ])
#     y_transforms = transforms.ToTensor()

#     db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train", img_size=args.img_size,
#                                norm_x_transform=x_transforms, norm_y_transform=y_transforms)
#     print(f"The length of train set is: {len(db_train)}")

#     def worker_init_fn(worker_id):
#         random.seed(args.seed + worker_id)

#     trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=args.num_workers,
#                              pin_memory=True, worker_init_fn=worker_init_fn)

#     db_test = Synapse_dataset(base_dir=args.test_path, split="test_vol", list_dir=args.list_dir,
#                               img_size=args.img_size)
#     testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)

#     if args.n_gpu > 1:
#         model = nn.DataParallel(model)

#     model.train()
#     logging.info("\n📌 **Computing Model Complexity:**")
#     with torch.cuda.device(0):
#         macs, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True, verbose=True)
#     logging.info(f"📌 MACs: {macs}")
#     logging.info(f"📌 Params: {params}")
#     print(f"MACs: {macs}")
#     print(f"Params: {params}")

#     # Loss functions
#     ce_loss = nn.CrossEntropyLoss()
#     dice_loss = DiceLoss(num_classes)
#     focal_loss = FocalLoss(alpha=0.20, gamma=1.0, reduction='mean')
#     loss_fn = HybridSegmentationLoss(num_classes=num_classes, alpha=0.6, gamma=2.0, edge_weight=1.0).cuda()

#     # Loss weights
#     focal_loss_weight = 0.2
#     dice_loss_weight = 0.7
#     ce_loss_weight = 0.2

#     optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
#     # === Initialize Discriminator ===
#     discriminator = MultiScalePatchDiscriminator(num_classes=num_classes).cuda()
#     optimizer_d = optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.5, 0.999))
#     adv_criterion = nn.BCEWithLogitsLoss()
#     writer = SummaryWriter(snapshot_path + '/log')
#     iter_num = 0
#     max_epoch = args.max_epochs
#     max_iterations = max_epoch * len(trainloader)
#     logging.info(f"{len(trainloader)} iterations per epoch. {max_iterations} max iterations")

#     best_performance = 0.0
#     dice_, hd95_ = [], []
#     iterator = tqdm(range(max_epoch), ncols=70)

#     for epoch_num in iterator:
#         for i_batch, sampled_batch in enumerate(trainloader):
#             image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
#             image_batch, label_batch = image_batch.cuda(), label_batch.squeeze(1).cuda()
#             outputs = model(image_batch)
            

#             if isinstance(outputs, list):  # Deep supervision
#                 supervision_weights = [0.1, 0.2, 0.3, 0.4]
#                 total_loss, total_dice, total_ce, total_edge = 0, 0, 0, 0

#                 for i, out in enumerate(outputs):
#                     #out = F.interpolate(out, size=label_batch.shape[1:], mode='bilinear', align_corners=False)
#                     edge_mask = create_edge_mask(label_batch)
                    
#                     # loss_ce = ce_loss(outputs, label_batch[:].long())
#                    # loss_dice = dice_loss(outputs, label_batch, softmax=True)
# #                   # loss = 0.4 * loss_ce + 0.6 * loss_dice
#                     #log_probs = F.log_softmax(out, dim=1)
#                     #ce = F.nll_loss(log_probs, label_batch.long(), ignore_index=255, reduction='mean')

#                     ce = ce_loss(out, label_batch[:].long(),reduction='mean')
#                     dice = dice_loss(out, label_batch, softmax=True)
#                     focal = focal_loss(out, label_batch)
#                     loss_lovasz = LovaszSoftmaxLoss()(outputs, label_batch)
#                     edge = loss_fn(out, label_batch, edge_mask)

#                     loss_i = (
#                         focal_loss_weight * focal +
#                         dice_loss_weight * dice +
#                         ce_loss_weight * ce +
#                         loss_lovasz*0.3 +
#                         0.3 * edge
#                     )
#                     weight = supervision_weights[i] if i < len(supervision_weights) else 1.0
#                     total_loss += (focal * focal_loss_weight + dice * dice_loss_weight + ce * ce_loss_weight + edge * 0.1 + 0.3* loss_lovasz) * weight
#                     total_ce += ce * weight
#                     total_dice += dice * weight
#                     total_edge += edge * weight
#                     total_weight += weight
                    

#                     if iter_num % 50 == 0 and i != 0:
#                         pred_vis = torch.argmax(torch.softmax(out, dim=1), dim=1, keepdim=True)
#                         writer.add_image(f'train/DS_Pred_{i}', pred_vis[1] * 50, iter_num)

#                 # loss = total_loss
#                 # loss_dice = total_dice
#                 # loss_ce = total_ce
#                 # loss_edge = total_edge
#                 loss = total_loss / total_weight
#                 loss_ce = total_ce / total_weight
#                 loss_dice = total_dice / total_weight
#                 loss_edge = loss_lovasz / total_weight
#                 outputs = outputs[0]

#             else:
#                 #outputs = F.interpolate(outputs, size=label_batch.shape[1:], mode='bilinear', align_corners=False)
#                 edge_mask = create_edge_mask(label_batch)

#                 loss_focal = focal_loss(outputs, label_batch)
#                 loss_dice = dice_loss(outputs, label_batch, softmax=True)
#                 loss_ce = ce_loss(outputs, label_batch[:].long())
#                 loss_edge = loss_fn(outputs, label_batch, edge_mask)
#                 loss_lov = LovaszSoftmaxLoss()(outputs, label_batch)
#                 loss = (
#                     focal_loss_weight * loss_focal +
#                     dice_loss_weight * loss_dice +
#                     ce_loss_weight * loss_ce +
#                     0.1 * loss_edge + 0.3*loss_lov
#                 )

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
#             for param_group in optimizer.param_groups:
#                 param_group['lr'] = lr_

#             iter_num += 1
#             writer.add_scalar('info/lr', lr_, iter_num)
#             writer.add_scalar('info/total_loss', loss, iter_num)
#             writer.add_scalar('info/loss_ce', loss_ce, iter_num)
#             writer.add_scalar('info/loss_dice', loss_dice, iter_num)
#             writer.add_scalar('info/loss_edge', loss_lov, iter_num)

#             logging.info(f"Iter {iter_num}: Loss {loss.item():.4f} |Edge {loss_lov.item():.4f} | CE {loss_ce.item():.4f} | Dice {loss_dice.item():.4f}")

#             if iter_num % 20 == 0:
#                 image = image_batch[1, 0:1, :, :]
#                 image = (image - image.min()) / (image.max() - image.min())
#                 writer.add_image('./train/Image', image, iter_num)
#                 pred = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
#                 writer.add_image('./train/Prediction', pred[1, ...] * 50, iter_num)
#                 labs = label_batch[1, ...].unsqueeze(0) * 50
#                 writer.add_image('./train/GroundTruth', labs, iter_num)

#         # Evaluation
#         if epoch_num >= int(max_epoch / 2) and (epoch_num + 1) % args.eval_interval == 0:
#             filename = f'{args.model_name}_epoch_{epoch_num}.pth'
#             save_path = os.path.join(snapshot_path, filename)
#             torch.save(model.state_dict(), save_path)
#             logging.info(f"✅ Saved model to {save_path}")

#             logging.info(f"Running Inference after Epoch {epoch_num}")
#             mean_dice, mean_hd95 = inference(model, testloader, args, test_save_path=test_save_path)
#             dice_.append(mean_dice)
#             hd95_.append(mean_hd95)
#             model.train()

#         if epoch_num >= max_epoch - 1:
#             filename = f'{args.model_name}_epoch_{epoch_num}.pth'
#             save_path = os.path.join(snapshot_path, filename)
#             torch.save(model.state_dict(), save_path)

#             if not (epoch_num + 1) % args.eval_interval == 0:
#                 logging.info(f"Final Evaluation at Epoch {epoch_num}")
#                 mean_dice, mean_hd95 = inference(model, testloader, args, test_save_path=test_save_path)
#                 dice_.append(mean_dice)
#                 hd95_.append(mean_hd95)
#                 model.train()

#             iterator.close()
#             break

#     plot_result(dice_, hd95_, snapshot_path, args)
#     writer.close()
#     return "Training Finished ✅"
# class MultiScalePatchDiscriminator(nn.Module):
#     def __init__(self, num_classes=9):
#         super(MultiScalePatchDiscriminator, self).__init__()
#         self.block = nn.Sequential(
#             nn.Conv2d(num_classes, 64, 4, stride=2, padding=1),
#             nn.LeakyReLU(0.2, inplace=True),

#             nn.Conv2d(64, 128, 4, stride=2, padding=1),
#             nn.BatchNorm2d(128),
#             nn.LeakyReLU(0.2, inplace=True),

#             nn.Conv2d(128, 256, 4, stride=2, padding=1),
#             nn.BatchNorm2d(256),
#             nn.LeakyReLU(0.2, inplace=True),

#             nn.Conv2d(256, 1, 4, padding=1)
#         )

#     def forward(self, x):
#         return self.block(x)


# class SliceFeatureExtractor(nn.Module):
#     def __init__(self, num_classes=9):
#         super(SliceFeatureExtractor, self).__init__()
#         self.features = nn.Sequential(
#             nn.Conv2d(num_classes, 64, kernel_size=4, stride=2, padding=1),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
#             nn.BatchNorm2d(128),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
#             nn.BatchNorm2d(256),
#             nn.LeakyReLU(0.2, inplace=True)
#         )
        
#     def forward(self, x):
#         # x shape: [B, num_classes, H, W]
#         out = self.features(x)
#         # Flatten the spatial dimensions
#         return out.view(out.size(0), -1)

# class RecurrentDiscriminator(nn.Module):
#     def __init__(self, num_classes=9, hidden_dim=512, num_layers=1, input_img_size=256):
#         super(RecurrentDiscriminator, self).__init__()
#         self.extractor = SliceFeatureExtractor(num_classes)
        
#         # Calculate the feature dimension after the conv layers.
#         # Assuming input images are input_img_size x input_img_size and three conv layers with stride=2.
#         # The spatial dimensions become: input_img_size / (2^3)
#         spatial_size = input_img_size // 8  # e.g., 256/8 = 32
#         feat_dim = 256 * spatial_size * spatial_size  # 256 is the number of output channels of last conv
        
#         self.lstm = nn.LSTM(feat_dim, hidden_dim, num_layers, batch_first=True)
#         self.classifier = nn.Linear(hidden_dim, 1)
    
#     def forward(self, x):
#         """
#         x: Tensor of shape [B, D, num_classes, H, W]
#            where B = batch size, D = number of slices
#         """
#         B, D, C, H, W = x.shape
#         features = []
#         for d in range(D):
#             # Process each slice independently using the 2D CNN extractor
#             feat = self.extractor(x[:, d])  # shape: [B, feat_dim]
#             features.append(feat)
        
#         # Stack to form a sequence: [B, D, feat_dim]
#         features = torch.stack(features, dim=1)
#         # Process the sequence with LSTM to capture inter-slice continuity
#         lstm_out, _ = self.lstm(features)  # lstm_out shape: [B, D, hidden_dim]
#         # Use the final LSTM output for classification
#         final_feature = lstm_out[:, -1]  # shape: [B, hidden_dim]
#         output = self.classifier(final_feature)  # shape: [B, 1]
#         return output
# def trainer_synapse(args, model, snapshot_path):
#     os.makedirs(os.path.join(snapshot_path, 'test'), exist_ok=True)
#     test_save_path = os.path.join(snapshot_path, 'test')

#     logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
#                         format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
#     logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
#     logging.info(str(args))

#     base_lr = args.base_lr
#     num_classes = args.num_classes
#     batch_size = args.batch_size * args.n_gpu

#     x_transforms = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize([0.5], [0.5])
#     ])
#     y_transforms = transforms.ToTensor()

#     db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train", img_size=args.img_size,
#                                norm_x_transform=x_transforms, norm_y_transform=y_transforms)
#     print(f"The length of train set is: {len(db_train)}")

#     def worker_init_fn(worker_id):
#         random.seed(args.seed + worker_id)

#     trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=args.num_workers,
#                              pin_memory=True, worker_init_fn=worker_init_fn)

#     db_test = Synapse_dataset(base_dir=args.test_path, split="test_vol", list_dir=args.list_dir,
#                               img_size=args.img_size)
#     testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)

#     if args.n_gpu > 1:
#         model = nn.DataParallel(model)

#     model.train()
#     logging.info("\n📌 **Computing Model Complexity:**")
#     with torch.cuda.device(0):
#         macs, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True, verbose=True)
#     logging.info(f"📌 MACs: {macs}")
#     logging.info(f"📌 Params: {params}")
#     print(f"MACs: {macs}")
#     print(f"Params: {params}")

#     # Loss functions
#     ce_loss = nn.CrossEntropyLoss()
#     dice_loss = DiceLoss(num_classes)
#     focal_loss = FocalLoss(alpha=0.20, gamma=1.0, reduction='mean')
#     loss_fn = HybridSegmentationLoss(num_classes=num_classes, alpha=0.6, gamma=2.0, edge_weight=1.0).cuda()

#     focal_loss_weight = 0.2
#     dice_loss_weight = 0.6
#     ce_loss_weight = 0.2

#     optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
#     writer = SummaryWriter(snapshot_path + '/log')

#     # === Initialize Discriminator ===
#     discriminator = MultiScalePatchDiscriminator(num_classes=num_classes).cuda()
#     optimizer_d = optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.5, 0.999))
#     adv_criterion = nn.BCEWithLogitsLoss()

#     iter_num = 0
#     max_epoch = args.max_epochs
#     max_iterations = max_epoch * len(trainloader)
#     logging.info(f"{len(trainloader)} iterations per epoch. {max_iterations} max iterations")

#     best_performance = 0.0
#     dice_, hd95_ = [], []
#     iterator = tqdm(range(max_epoch), ncols=70)

#     for epoch_num in iterator:
#         for i_batch, sampled_batch in enumerate(trainloader):
#             image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
#             image_batch, label_batch = image_batch.cuda(), label_batch.squeeze(1).cuda()
#             outputs = model(image_batch)

#             edge_mask = create_edge_mask(label_batch)

#             loss_focal = focal_loss(outputs, label_batch)
#             loss_dice = dice_loss(outputs, label_batch, softmax=True)
#             loss_ce = ce_loss(outputs, label_batch[:].long())
#             loss_edge = loss_fn(outputs, label_batch, edge_mask)

#             # === Discriminator Training ===
#             with torch.no_grad():
#                 pred_fake = discriminator(torch.softmax(outputs.detach(), dim=1))
#             valid_label = torch.zeros_like(pred_fake)
#             loss_d_fake = adv_criterion(pred_fake, valid_label)
#             label_batch=label_batch.long()
#             gt_one_hot = F.one_hot(label_batch, num_classes=num_classes).permute(0, 3, 1, 2).float()
#             pred_real = discriminator(gt_one_hot)
#             real_label = torch.ones_like(pred_real)
#             loss_d_real = adv_criterion(pred_real, real_label)

#             loss_d = 0.5 * (loss_d_fake + loss_d_real)
#             optimizer_d.zero_grad()
#             loss_d.backward()
#             optimizer_d.step()

#             # === Generator Adversarial Loss ===
#             pred_fake_for_g = discriminator(torch.softmax(outputs, dim=1))
#             loss_g_adv = adv_criterion(pred_fake_for_g, real_label)

#             loss = (
#                 focal_loss_weight * loss_focal +
#                 dice_loss_weight * loss_dice +
#                 ce_loss_weight * loss_ce +
#                 0.3 * loss_edge +
#                 0.01 * loss_g_adv
#             )

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
#             for param_group in optimizer.param_groups:
#                 param_group['lr'] = lr_

#             iter_num += 1
#             writer.add_scalar('info/lr', lr_, iter_num)
#             writer.add_scalar('info/total_loss', loss, iter_num)
#             writer.add_scalar('info/loss_ce', loss_ce, iter_num)
#             writer.add_scalar('info/loss_dice', loss_dice, iter_num)
#             writer.add_scalar('info/loss_edge', loss_edge, iter_num)
#             writer.add_scalar('info/loss_adv', loss_g_adv, iter_num)
#             writer.add_scalar('info/loss_d', loss_d, iter_num)

#             logging.info(f"Iter {iter_num}: Loss {loss.item():.4f} |Adv {loss_g_adv.item():.4f} |Edge {loss_edge.item():.4f} | CE {loss_ce.item():.4f} | Dice {loss_dice.item():.4f}")

#         # Evaluation
#         if epoch_num >= int(max_epoch / 2) and (epoch_num + 1) % args.eval_interval == 0:
#             filename = f'{args.model_name}_epoch_{epoch_num}.pth'
#             save_path = os.path.join(snapshot_path, filename)
#             torch.save(model.state_dict(), save_path)
#             logging.info(f"✅ Saved model to {save_path}")

#             logging.info(f"Running Inference after Epoch {epoch_num}")
#             mean_dice, mean_hd95 = inference(model, testloader, args, test_save_path=test_save_path)
#             dice_.append(mean_dice)
#             hd95_.append(mean_hd95)
#             model.train()

#         if epoch_num >= max_epoch - 1:
#             filename = f'{args.model_name}_epoch_{epoch_num}.pth'
#             save_path = os.path.join(snapshot_path, filename)
#             torch.save(model.state_dict(), save_path)

#             if not (epoch_num + 1) % args.eval_interval == 0:
#                 logging.info(f"Final Evaluation at Epoch {epoch_num}")
#                 mean_dice, mean_hd95 = inference(model, testloader, args, test_save_path=test_save_path)
#                 dice_.append(mean_dice)
#                 hd95_.append(mean_hd95)
#                 model.train()

#             iterator.close()
#             break

#     plot_result(dice_, hd95_, snapshot_path, args)
#     writer.close()
#     return "Training Finished ✅"
# def trainer_synapse(args, model, snapshot_path):
#     os.makedirs(os.path.join(snapshot_path, 'test'), exist_ok=True)
#     test_save_path = os.path.join(snapshot_path, 'test')

#     logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
#                         format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
#     logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
#     logging.info(str(args))

#     base_lr = args.base_lr
#     num_classes = args.num_classes
#     batch_size = args.batch_size * args.n_gpu

#     x_transforms = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize([0.5], [0.5])
#     ])
#     y_transforms = transforms.ToTensor()

#     db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train", img_size=args.img_size,
#                                norm_x_transform=x_transforms, norm_y_transform=y_transforms)
#     print(f"The length of train set is: {len(db_train)}")

#     def worker_init_fn(worker_id):
#         random.seed(args.seed + worker_id)

#     trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=args.num_workers,
#                              pin_memory=True, worker_init_fn=worker_init_fn)

#     db_test = Synapse_dataset(base_dir=args.test_path, split="test_vol", list_dir=args.list_dir,
#                               img_size=args.img_size)
#     testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)

#     if args.n_gpu > 1:
#         model = nn.DataParallel(model)

#     model.train()
#     logging.info("\n📌 **Computing Model Complexity:**")
#     with torch.cuda.device(0):
#         macs, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True, verbose=True)
#     logging.info(f"📌 MACs: {macs}")
#     logging.info(f"📌 Params: {params}")
#     print(f"MACs: {macs}")
#     print(f"Params: {params}")

#     # Loss functions
#     ce_loss = nn.CrossEntropyLoss()
#     dice_loss = DiceLoss(num_classes)
#     focal_loss = FocalLoss(alpha=0.20, gamma=1.0, reduction='mean')
#     loss_fn = HybridSegmentationLoss(num_classes=num_classes, alpha=0.6, gamma=2.0, edge_weight=1.0).cuda()
    
#     focal_loss_weight = 0.2
#     dice_loss_weight = 0.6
#     ce_loss_weight = 0.2

#     optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
#     writer = SummaryWriter(snapshot_path + '/log')

#     discriminator = MultiScalePatchDiscriminator(num_classes=num_classes).cuda()
#     optimizer_d = optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.5, 0.999))
#     adv_criterion = nn.BCEWithLogitsLoss()

#     max_epoch = args.max_epochs
#     max_iterations = max_epoch * len(trainloader)
#     logging.info(f"{len(trainloader)} iterations per epoch. {max_iterations} max iterations")

#     best_performance = 0.0
#     dice_, hd95_ = [], []

#     # === Resume Training if Checkpoint Exists ===
#     start_epoch = 0
#     iter_num = 0
#     if args.resume:
#         checkpoint_path = os.path.join(snapshot_path, 'checkpoint_latest.pth')
#         if os.path.exists(checkpoint_path):
#             checkpoint = torch.load(checkpoint_path)
#             model.load_state_dict(checkpoint['model_state_dict'])
#             optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#             discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
#             optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
#             start_epoch = checkpoint['epoch'] + 1
#             iter_num = checkpoint['iter_num']
#             best_performance = checkpoint.get('best_dice', 0.0)
#             logging.info(f"🔁 Resumed from checkpoint at epoch {start_epoch}, iter {iter_num}")
#         else:
#             logging.warning("⚠️ Resume requested, but no checkpoint found.")

#     iterator = tqdm(range(start_epoch, max_epoch), ncols=70)

#     for epoch_num in iterator:
#         for i_batch, sampled_batch in enumerate(trainloader):
#             image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
#             image_batch, label_batch = image_batch.cuda(), label_batch.squeeze(1).cuda()
#             outputs = model(image_batch)

#             edge_mask = create_edge_mask(label_batch)

#             loss_focal = focal_loss(outputs, label_batch)
#             loss_dice = dice_loss(outputs, label_batch, softmax=True)
#             loss_ce = ce_loss(outputs, label_batch.long())
#             loss_edge = loss_fn(outputs, label_batch, edge_mask)
#             loss_lov = LovaszSoftmaxLoss()(outputs, label_batch)
#             # === Discriminator Training ===
#             with torch.no_grad():
#                 pred_fake = discriminator(torch.softmax(outputs.detach(), dim=1))
#             valid_label = torch.zeros_like(pred_fake)
#             loss_d_fake = adv_criterion(pred_fake, valid_label)

#             gt_one_hot = F.one_hot(label_batch.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()
#             pred_real = discriminator(gt_one_hot)
#             real_label = torch.ones_like(pred_real)
#             loss_d_real = adv_criterion(pred_real, real_label)

#             loss_d = 0.5 * (loss_d_fake + loss_d_real)
#             optimizer_d.zero_grad()
#             loss_d.backward()
#             optimizer_d.step()

#             # === Generator Adversarial Loss ===
#             pred_fake_for_g = discriminator(torch.softmax(outputs, dim=1))
#             loss_g_adv = adv_criterion(pred_fake_for_g, real_label)

#             loss = (
#                 focal_loss_weight * loss_focal +
#                 dice_loss_weight * loss_dice +
#                 ce_loss_weight * loss_ce +
#                 0.3 * loss_edge +
#                 0.01 * loss_g_adv +
#                 0.01 *loss_lov
#             )

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
#             for param_group in optimizer.param_groups:
#                 param_group['lr'] = lr_

#             iter_num += 1
#             writer.add_scalar('info/lr', lr_, iter_num)
#             writer.add_scalar('info/total_loss', loss, iter_num)
#             writer.add_scalar('info/loss_ce', loss_ce, iter_num)
#             writer.add_scalar('info/loss_dice', loss_dice, iter_num)
#             writer.add_scalar('info/loss_edge', loss_edge, iter_num)
#             writer.add_scalar('info/loss_adv', loss_g_adv, iter_num)
#             writer.add_scalar('info/loss_d', loss_d, iter_num)
#             writer.add_scalar('info/loss_lov',loss_lov,iter_num)

#             logging.info(f"Iter {iter_num}: Loss {loss.item():.4f} |Adv {loss_g_adv.item():.4f} |Edge {loss_edge.item():.4f} | CE {loss_ce.item():.4f} |Sl {loss_lov.item():.4f} | Dice {loss_dice.item():.4f}")

#         # Evaluation
#         if epoch_num >= int(max_epoch / 2) and (epoch_num + 1) % args.eval_interval == 0:
#             filename = f'{args.model_name}_epoch_{epoch_num}.pth'
#             save_path = os.path.join(snapshot_path, filename)
#             torch.save(model.state_dict(), save_path)
#             logging.info(f"✅ Saved model to {save_path}")

#             logging.info(f"Running Inference after Epoch {epoch_num}")
#             mean_dice, mean_hd95 = inference(model, testloader, args, test_save_path=test_save_path)
#             dice_.append(mean_dice)
#             hd95_.append(mean_hd95)
#             model.train()

#         # === Save Checkpoint ===
#         checkpoint = {
#             'epoch': epoch_num,
#             'iter_num': iter_num,
#             'model_state_dict': model.state_dict(),
#             'optimizer_state_dict': optimizer.state_dict(),
#             'discriminator_state_dict': discriminator.state_dict(),
#             'optimizer_d_state_dict': optimizer_d.state_dict(),
#             'best_dice': best_performance,
#         }
#         torch.save(checkpoint, os.path.join(snapshot_path, 'checkpoint_latest.pth'))

#         if epoch_num >= max_epoch - 1:
#             filename = f'{args.model_name}_epoch_{epoch_num}.pth'
#             save_path = os.path.join(snapshot_path, filename)
#             torch.save(model.state_dict(), save_path)

#             if not (epoch_num + 1) % args.eval_interval == 0:
#                 logging.info(f"Final Evaluation at Epoch {epoch_num}")
#                 mean_dice, mean_hd95 = inference(model, testloader, args, test_save_path=test_save_path)
#                 dice_.append(mean_dice)
#                 hd95_.append(mean_hd95)
#                 model.train()

#             iterator.close()
#             break

#     plot_result(dice_, hd95_, snapshot_path, args)
#     writer.close()
#     return "Training Finished ✅"

