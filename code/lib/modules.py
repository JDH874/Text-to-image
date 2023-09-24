import os, sys
from pyexpat import features
import os.path as osp
import time
import random
import datetime
import argparse
from scipy import linalg
import numpy as np
from PIL import Image
from tqdm import tqdm, trange
from torch.autograd import Variable
import clip                                            
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torchvision.utils import make_grid
from lib.utils import transf_to_CLIP_input, dummy_context_mgr
from lib.utils import mkdir_p, get_rank
from lib.datasets import prepare_data

from models.inception import InceptionV3
from torch.nn.functional import adaptive_avg_pool2d
import torch.distributed as dist


############   GAN   ############
def train(dataloader, netG, netD, netC, text_encoder, image_encoder, optimizerG, optimizerD, scaler_G, scaler_D, args):
    batch_size = args.batch_size
    device = args.device
    epoch = args.current_epoch
    max_epoch = args.max_epoch
    z_dim = args.z_dim
    netG, netD, netC, image_encoder = netG.train(), netD.train(), netC.train(), image_encoder.train()
    if (args.multi_gpus==True) and (get_rank() != 0):
        None
    else:
        loop = tqdm(total=len(dataloader))
    for step, data in enumerate(dataloader, 0):
        ##############
        # Train D  
        ##############
        optimizerD.zero_grad()
        with torch.cuda.amp.autocast() if args.mixed_precision else dummy_context_mgr() as mpc:
            # prepare_data
            real, captions, CLIP_tokens, sent_emb, words_embs, asp_emb1, asp_emb2, asp_emb3, keys = \
                                                                      prepare_data(data, text_encoder, device)          #1
            real = real.requires_grad_()
            sent_emb = sent_emb.requires_grad_()
            words_embs = words_embs.requires_grad_()
            asp_emb1 = asp_emb1.requires_grad_()
            asp_emb2 = asp_emb2.requires_grad_()
            asp_emb3 = asp_emb3.requires_grad_()

            # predict real
            CLIP_real,real_emb = image_encoder(real)
            real_feats, real_feats_pooled = netD(CLIP_real)
            
            pred_real, errD_real = predict_loss(netC, real_feats, sent_emb, negtive=False)

            # predict mismatch
            mis_sent_emb = torch.cat((sent_emb[1:], sent_emb[0:1]), dim=0).detach()
            _, errD_mis = predict_loss(netC, real_feats, mis_sent_emb, negtive=True)
            # synthesize fake images
            noise = torch.randn(batch_size, z_dim).to(device)
            fake = netG(noise, sent_emb, asp_emb1, asp_emb2, asp_emb3, words_embs)
            CLIP_fake, fake_emb = image_encoder(fake)
            fake_feats, fake_feats_pooled = netD(CLIP_fake.detach())
            _, errD_fake = predict_loss(netC, fake_feats, sent_emb, negtive=True)


            #contrastive loss
            d_sent_cl_loss = contrastive_loss(real_feats_pooled, sent_emb.detach(), device=device)
            d_asp1_cl_loss = contrastive_loss(real_feats_pooled, asp_emb1.detach(), device=device)
            d_asp2_cl_loss = contrastive_loss(real_feats_pooled, asp_emb2.detach(), device=device)
            d_asp3_cl_loss = contrastive_loss(real_feats_pooled, asp_emb3.detach(), device=device)
            d_asp_cl_loss = (d_asp1_cl_loss + d_asp2_cl_loss + d_asp3_cl_loss)/3
            loss_Dc = d_asp_cl_loss + d_sent_cl_loss\
            
            #word loss
            aspects_prob = torch.cosine_similarity(real_feats_pooled.unsqueeze(1), words_embs, dim=2)
            D_loss_w = (1 - aspects_prob).mean()

        # MA-GP
        if args.mixed_precision:
            errD_MAGP = MA_GP_MP(CLIP_real, sent_emb, pred_real, scaler_D)
            
        else:
            errD_MAGP = MA_GP_FP32(CLIP_real, sent_emb, pred_real)

        # whole D loss
        with torch.cuda.amp.autocast() if args.mixed_precision else dummy_context_mgr() as mpc:
            errD_s = errD_real + (errD_fake + errD_mis)/2.0 + errD_MAGP + 0.1 * loss_Dc + 0.2 * D_loss_w
            errD = errD_s
            
        # update D
        if args.mixed_precision:
            scaler_D.scale(errD).backward()
            scaler_D.step(optimizerD)
            scaler_D.update()
            if scaler_D.get_scale()<args.scaler_min:
                scaler_D.update(16384.0)
        else:
            errD.backward()
            optimizerD.step()
        ##############
        # Train G  
        ##############
        optimizerG.zero_grad()
        with torch.cuda.amp.autocast() if args.mixed_precision else dummy_context_mgr() as mpc:
            
            fake_feats, fake_feats_pooled = netD(CLIP_fake)
            output = netC(fake_feats, sent_emb)
            G_loss = -output.mean()
            text_img_sim = torch.cosine_similarity(fake_emb, sent_emb).mean()
            aspects_prob = torch.cosine_similarity(fake_feats_pooled.unsqueeze(1), words_embs, dim=2)
            G_loss_w = (1 - aspects_prob).mean()
            
            #contrastive loss
            g_sent_cl_loss = contrastive_loss(fake_feats_pooled, sent_emb.detach(), device=device)
            g_asp1_cl_loss = contrastive_loss(fake_feats_pooled, asp_emb1.detach(), device=device)
            g_asp2_cl_loss = contrastive_loss(fake_feats_pooled, asp_emb2.detach(), device=device)
            g_asp3_cl_loss = contrastive_loss(fake_feats_pooled, asp_emb3.detach(), device=device)
            g_asp_cl_loss = (g_asp1_cl_loss + g_asp2_cl_loss + g_asp3_cl_loss)/3
            loss_Gc = g_sent_cl_loss + g_asp_cl_loss 
            
            errG = G_loss - args.sim_w*text_img_sim + 0.1 * loss_Gc + 0.2 * G_loss_w

        if args.mixed_precision:
            scaler_G.scale(errG).backward()
            scaler_G.step(optimizerG)
            scaler_G.update()
            if scaler_G.get_scale()<args.scaler_min:
                scaler_G.update(16384.0)
        else:
            errG.backward()
            optimizerG.step()
        # update loop information
        if (args.multi_gpus==True) and (get_rank() != 0):
            None
        else:
            loop.update(1)
            loop.set_description(f'Train Epoch [{epoch}/{max_epoch}]')
            loop.set_postfix()
    if (args.multi_gpus==True) and (get_rank() != 0):
        None
    else:
        loop.close()

# contrastive loss
def l2_normalize(x, dim=None, epsilon=1e-12):
    square_sum = torch.sum(torch.square(x), dim=dim, keepdims=True)
    x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.ones_like(square_sum) * epsilon))
    return torch.multiply(x, x_inv_norm)

def contrastive_loss(image_feat, cond_feat, l2_norm=True, temperature=0.1, device="cpu"):
    """Calculates contrastive loss."""
    if l2_norm:
        image_feat = l2_normalize(image_feat, -1)
        cond_feat = l2_normalize(cond_feat, -1)
    local_batch_size = image_feat.shape[0]
    image_feat_large = image_feat
    cond_feat_large = cond_feat
    labels = torch.arange(local_batch_size).to(device)
    # labels_onehot = F.one_hot(labels, local_batch_size)
    logits_img2cond = torch.matmul(image_feat,
                                 cond_feat_large.t()) / temperature
    logits_cond2img = torch.matmul(cond_feat,
                                 image_feat_large.t()) / temperature
    loss_img2cond = F.cross_entropy(logits_img2cond, labels)
    loss_cond2img = F.cross_entropy(logits_cond2img, labels)
    loss_img2cond = torch.mean(loss_img2cond)
    loss_cond2img = torch.mean(loss_cond2img)
    loss = loss_img2cond + loss_cond2img
    return loss
        
        
#clip encoder  ###
def clip_text_embedding(text, clip_model, device):
    text = clip.tokenize(text).to(device)
    text_features = clip_model.encode_text(text)
    return text, text_features.float()


def clip_image_embedding(image, clip_model, device):
    image = image.to(device)
    image_features = clip_model.encode_image(image)
    return image_features.float()


def load_clip(device):
    clip_model, preprocess = clip.load('ViT-B/32', device)
    clip_pool = torch.nn.AdaptiveAvgPool2d((224, 224))
    clip_trans = transforms.Compose(
        [transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])
    for param in clip_model.parameters():
        param.requires_grad = False
    return clip_model, clip_pool, clip_trans


def test(dataloader, text_encoder, netG, PTM, device, m1, s1, epoch, max_epoch, times, z_dim, batch_size):
    FID, TI_sim = calculate_FID_CLIP_sim(dataloader, text_encoder, netG, PTM, device, m1, s1, epoch, max_epoch, times, z_dim, batch_size)
    return FID, TI_sim


def save_model(netG, netD, netC, optG, optD, epoch, multi_gpus, step, save_path):
    if (multi_gpus==True) and (get_rank() != 0):
        None
    else:
        state = {'model': {'netG': netG.state_dict(), 'netD': netD.state_dict(), 'netC': netC.state_dict()}, \
                'optimizers': {'optimizer_G': optG.state_dict(), 'optimizer_D': optD.state_dict()},\
                'epoch': epoch}
        torch.save(state, '%s/state_epoch_%03d_%03d.pth' % (save_path, epoch, step))


#########   MAGP   ########
def MA_GP_MP(img, sent, out, scaler):
    grads = torch.autograd.grad(outputs=scaler.scale(out),
                            inputs=(img, sent),
                            grad_outputs=torch.ones_like(out),
                            retain_graph=True,
                            create_graph=True,
                            only_inputs=True)
    inv_scale = 1./(scaler.get_scale()+float("1e-8"))
    #inv_scale = 1./scaler.get_scale()
    grads = [grad * inv_scale for grad in grads]
    with torch.cuda.amp.autocast():
        grad0 = grads[0].view(grads[0].size(0), -1)
        grad1 = grads[1].view(grads[1].size(0), -1)
        grad = torch.cat((grad0,grad1),dim=1)                        
        grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
        d_loss_gp =  2.0 * torch.mean((grad_l2norm) ** 6)
    return d_loss_gp


def MA_GP_FP32(img, sent, out):
    grads = torch.autograd.grad(outputs=out,
                            inputs=(img, sent),
                            grad_outputs=torch.ones(out.size()).cuda(),
                            retain_graph=True,
                            create_graph=True,
                            only_inputs=True)
    grad0 = grads[0].view(grads[0].size(0), -1)
    grad1 = grads[1].view(grads[1].size(0), -1)
    grad = torch.cat((grad0,grad1),dim=1)                        
    grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
    d_loss_gp =  2.0 * torch.mean((grad_l2norm) ** 6)
    return d_loss_gp


def sample(dataloader, netG, text_encoder, save_dir, device, multi_gpus, z_dim, stamp):
    netG.eval()
    for step, data in enumerate(dataloader, 0):
        ######################################################
        # (1) Prepare_data
        ######################################################
        real, captions, CLIP_tokens, sent_emb, words_embs, asp_emb1, asp_emb2, asp_emb3, keys = \
                                         prepare_data(data, text_encoder, device)                                       #1
        ######################################################
        # (2) Generate fake images
        ######################################################
        batch_size = sent_emb.size(0)
        with torch.no_grad():
            noise = torch.randn(batch_size, z_dim).to(device)
            fake_imgs = netG(noise, sent_emb, asp_emb1, asp_emb2, asp_emb3, words_embs, eval=True).float()              #1
            fake_imgs = torch.clamp(fake_imgs, -1., 1.)
            if multi_gpus==True:
                batch_img_name = 'step_%04d.png'%(step)
                batch_img_save_dir  = osp.join(save_dir, 'batch', str('gpu%d'%(get_rank())), 'imgs')
                batch_img_save_name = osp.join(batch_img_save_dir, batch_img_name)
                batch_txt_name = 'step_%04d.txt'%(step)
                batch_txt_save_dir  = osp.join(save_dir, 'batch', str('gpu%d'%(get_rank())), 'txts')
                batch_txt_save_name = osp.join(batch_txt_save_dir, batch_txt_name)
            else:
                batch_img_name = 'step_%04d.png'%(step)
                batch_img_save_dir  = osp.join(save_dir, 'batch', 'imgs')
                batch_img_save_name = osp.join(batch_img_save_dir, batch_img_name)
                batch_txt_name = 'step_%04d.txt'%(step)
                batch_txt_save_dir  = osp.join(save_dir, 'batch', 'txts')
                batch_txt_save_name = osp.join(batch_txt_save_dir, batch_txt_name)
            mkdir_p(batch_img_save_dir)
            vutils.save_image(fake_imgs.data, batch_img_save_name, nrow=8, value_range=(-1, 1), normalize=True)
            mkdir_p(batch_txt_save_dir)
            txt = open(batch_txt_save_name,'w')
            for cap in captions:
                txt.write(cap+'\n')
            txt.close()
            for j in range(batch_size):
                im = fake_imgs[j].data.cpu().numpy()
                # [-1, 1] --> [0, 255]
                im = (im + 1.0) * 127.5
                im = im.astype(np.uint8)
                im = np.transpose(im, (1, 2, 0))
                im = Image.fromarray(im)
                ######################################################
                # (3) Save fake images
                ######################################################      
                if multi_gpus==True:
                    single_img_name = 'batch_%04d.png'%(j)
                    single_img_save_dir  = osp.join(save_dir, 'single', str('gpu%d'%(get_rank())), 'step%04d'%(step))
                    single_img_save_name = osp.join(single_img_save_dir, single_img_name)
                else:
                    single_img_name = 'step_%04d.png'%(step)
                    single_img_save_dir  = osp.join(save_dir, 'single', 'step%04d'%(step))
                    single_img_save_name = osp.join(single_img_save_dir, single_img_name)   
                mkdir_p(single_img_save_dir)   
                im.save(single_img_save_name)
        if (multi_gpus==True) and (get_rank() != 0):
            None
        else:
            print('Step: %d' % (step))


def calculate_FID_CLIP_sim(dataloader, text_encoder, netG, CLIP, device, m1, s1, epoch, max_epoch, times, z_dim, batch_size):
    """ Calculates the FID """
    clip_cos = torch.FloatTensor([0.0]).to(device)
    # prepare Inception V3
    dims = 2048
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx])
    model.to(device)
    model.eval()
    netG.eval()
    norm = transforms.Compose([
        transforms.Normalize((-1, -1, -1), (2, 2, 2)),
        transforms.Resize((299, 299)),
        ])
    n_gpu = dist.get_world_size()
    dl_length = dataloader.__len__()
    imgs_num = dl_length * n_gpu * batch_size * times
    pred_arr = np.empty((imgs_num, dims))
    if (n_gpu!=1) and (get_rank() != 0):
        None
    else:
        loop = tqdm(total=int(dl_length*times))
    for time in range(times):
        for i, data in enumerate(dataloader):
            start = i * batch_size * n_gpu + time * dl_length * n_gpu * batch_size
            end = start + batch_size * n_gpu
            ######################################################
            # (1) Prepare_data
            ######################################################
            imgs, captions, CLIP_tokens, sent_emb, words_embs, asp_emb1, asp_emb2, asp_emb3, keys = \
                                                             prepare_data(data, text_encoder, device)               
            ######################################################
            # (2) Generate fake images
            ######################################################
            batch_size = sent_emb.size(0)
            netG.eval()
            with torch.no_grad():
                noise = torch.randn(batch_size, z_dim).to(device)
                fake_imgs = netG(noise,sent_emb,asp_emb1, asp_emb2, asp_emb3,words_embs,eval=True).float()             

                if save_imgs == True:                                                          
                    save_single_imgs(fake_imgs, save_dir, time, dl_length, i, batch_size)        

                fake_imgs = torch.clamp(fake_imgs, -1., 1.)
                fake_imgs = torch.nan_to_num(fake_imgs, nan=-1.0, posinf=1.0, neginf=-1.0)
                clip_sim = calc_clip_sim(CLIP, fake_imgs, CLIP_tokens, device)
                clip_cos = clip_cos + clip_sim
                fake = norm(fake_imgs)
                pred = model(fake)[0]
                if pred.shape[2] != 1 or pred.shape[3] != 1:
                    pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
                # concat pred from multi GPUs
                output = list(torch.empty_like(pred) for _ in range(n_gpu))
                dist.all_gather(output, pred)
                pred_all = torch.cat(output, dim=0).squeeze(-1).squeeze(-1)
                pred_arr[start:end] = pred_all.cpu().data.numpy()
            # update loop information
            if (n_gpu!=1) and (get_rank() != 0):
                None
            else:
                loop.update(1)
                if epoch==-1:
                    loop.set_description('Evaluating]')
                else:
                    loop.set_description(f'Eval Epoch [{epoch}/{max_epoch}]')
                loop.set_postfix()
    if (n_gpu!=1) and (get_rank() != 0):
        None
    else:
        loop.close()
    # CLIP-score
    CLIP_score_gather = list(torch.empty_like(clip_cos) for _ in range(n_gpu))
    dist.all_gather(CLIP_score_gather, clip_cos)
    clip_score = torch.cat(CLIP_score_gather, dim=0).mean().item()/(dl_length*times)
    # FID
    m2 = np.mean(pred_arr, axis=0)
    s2 = np.cov(pred_arr, rowvar=False)
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)
    return fid_value,clip_score

def save_single_imgs(imgs, save_dir, time, dl_len, batch_n, batch_size):
    for j in range(batch_size):
        folder = save_dir
        if not os.path.isdir(folder):
            mkdir_p(folder)
        im = imgs[j].data.cpu().numpy()
        # [-1, 1] --> [0, 255]
        im = (im + 1.0) * 127.5
        im = im.astype(np.uint8)
        im = np.transpose(im, (1, 2, 0))
        im = Image.fromarray(im)
        filename = 'imgs_n%06d_gpu%1d.png'%(time*dl_len+batch_size*batch_n+j, get_rank())
        fullpath = osp.join(folder, filename)
        im.save(fullpath)


def calc_clip_sim(clip, fake, caps_clip, device):
    # Calculate features
    fake = transf_to_CLIP_input(fake)
    fake_features = clip.encode_image(fake)
    text_features = clip.encode_text(caps_clip)
    text_img_sim = torch.cosine_similarity(fake_features, text_features).mean()
    return text_img_sim


def sample_one_batch(noise, sent, asp1, asp2, asp3, words, netG, multi_gpus, epoch, img_save_dir, writer):
    if (multi_gpus==True) and (get_rank() != 0):
        None
    else:
        netG.eval()
        with torch.no_grad():
            B = noise.size(0)
            fixed_results_train = generate_samples(noise[:B//2], sent[:B//2], asp1[:B//2], asp2[:B//2],
                                                   asp3[:B//2], words[:B//2], netG).cpu()                               
            torch.cuda.empty_cache()
            fixed_results_test = generate_samples(noise[B//2:], sent[B//2:], asp1[:B//2], asp2[:B//2],
                                                   asp3[:B//2], words[:B//2], netG).cpu()                               
            torch.cuda.empty_cache()
            fixed_results = torch.cat((fixed_results_train, fixed_results_test), dim=0)
        img_name = 'samples_epoch_%03d.png'%(epoch)
        img_save_path = osp.join(img_save_dir, img_name)
        vutils.save_image(fixed_results.data, img_save_path, nrow=8, value_range=(-1, 1), normalize=True)


def generate_samples(noise, caption, asp1, asp2, asp3, words, model):                                                   
    with torch.no_grad():
        fake = model(noise, caption, asp1, asp2, asp3, words, eval=True)                                                
    return fake


def predict_loss(predictor, img_feature, text_feature, negtive):
    output = predictor(img_feature, text_feature)
    err = hinge_loss(output, negtive)
    return output,err


def hinge_loss(output, negtive):
    if negtive==False:
        err = torch.mean(F.relu(1. - output))
    else:
        err = torch.mean(F.relu(1. + output))
    return err


def logit_loss(output, negtive):
    batch_size = output.size(0)
    real_labels = torch.FloatTensor(batch_size,1).fill_(1).to(output.device)
    fake_labels = torch.FloatTensor(batch_size,1).fill_(0).to(output.device)
    output = nn.Sigmoid()(output)
    if negtive==False:
        err = nn.BCELoss()(output, real_labels)
    else:
        err = nn.BCELoss()(output, fake_labels)
    return err


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2
    '''
    print('&'*20)
    print(sigma1)#, sigma1.type())
    print('&'*20)
    print(sigma2)#, sigma2.type())
    '''
    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)
