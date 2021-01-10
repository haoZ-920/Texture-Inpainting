import os
import time
import datetime
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

import network
import train_dataset
import utils

from Render import Render

def WGAN_trainer(opt):
    # ----------------------------------------
    #      Initialize training parameters
    # ----------------------------------------
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    # cudnn benchmark accelerates the network
    cudnn.benchmark = opt.cudnn_benchmark

    # configurations
    save_folder = opt.save_path
    sample_folder = opt.sample_path
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    if not os.path.exists(sample_folder):
        os.makedirs(sample_folder)

    # Build networks
    generator = utils.create_generator(opt)
    discriminator = utils.create_discriminator(opt)
    perceptualnet = utils.create_perceptualnet()

    # Loss functions
    L1Loss = nn.L1Loss()
    MSELoss = nn.MSELoss()

    # Optimizers
    optimizer_g = torch.optim.Adam(generator.parameters(), lr = opt.lr_g, betas = (opt.b1, opt.b2), weight_decay = opt.weight_decay)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr = opt.lr_d, betas = (opt.b1, opt.b2), weight_decay = opt.weight_decay)

    # Learning rate decrease
    def adjust_learning_rate(lr_in, optimizer, epoch, opt):
        """Set the learning rate to the initial LR decayed by "lr_decrease_factor" every "lr_decrease_epoch" epochs"""
        lr = lr_in * (opt.lr_decrease_factor ** (epoch // opt.lr_decrease_epoch))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    
    # Save the two-stage generator model
    def save_model_generator(net, epoch, opt):
        """Save the model at "checkpoint_interval" and its multiple"""
        model_name = 'deepfillv2_WGAN_G_epoch%d_batchsize%d.pth' % (epoch, opt.batch_size)
        model_name = os.path.join(save_folder, model_name)
        if opt.multi_gpu == True:
            if epoch % opt.checkpoint_interval == 0:
                torch.save(net.module.state_dict(), model_name)
                print('The trained model is successfully saved at epoch %d' % (epoch))
        else:
            if epoch % opt.checkpoint_interval == 0:
                torch.save(net.state_dict(), model_name)
                print('The trained model is successfully saved at epoch %d' % (epoch))
                
    # Save the dicriminator model
    def save_model_discriminator(net, epoch, opt):
        """Save the model at "checkpoint_interval" and its multiple"""
        model_name = 'deepfillv2_WGAN_D_epoch%d_batchsize%d.pth' % (epoch, opt.batch_size)
        model_name = os.path.join(save_folder, model_name)
        if opt.multi_gpu == True:
            if epoch % opt.checkpoint_interval == 0:
                torch.save(net.module.state_dict(), model_name)
                print('The trained model is successfully saved at epoch %d' % (epoch))
        else:
            if epoch % opt.checkpoint_interval == 0:
                torch.save(net.state_dict(), model_name)
                print('The trained model is successfully saved at epoch %d' % (epoch))
                
    # load the model
    def load_model(net, epoch, opt, type='G'):
        """Save the model at "checkpoint_interval" and its multiple"""
        if type == 'G':
            model_name = 'deepfillv2_WGAN_G_epoch%d_batchsize%d.pth' % (epoch, opt.batch_size)
        else:
            model_name = 'deepfillv2_WGAN_D_epoch%d_batchsize%d.pth' % (epoch, opt.batch_size)
        model_name = os.path.join(save_folder, model_name)
        pretrained_dict = torch.load(model_name)
        net.load_state_dict(pretrained_dict)

    if opt.resume:
        load_model(generator, opt.resume_epoch, opt, type='G')
        load_model(discriminator, opt.resume_epoch, opt, type='D')
        print('--------------------Pretrained Models are Loaded--------------------')
        
    # To device
    if opt.multi_gpu == True:
        generator = nn.DataParallel(generator)
        discriminator = nn.DataParallel(discriminator)
        perceptualnet = nn.DataParallel(perceptualnet)
        generator = generator.cuda()
        discriminator = discriminator.cuda()
        perceptualnet = perceptualnet.cuda()
    else:
        generator = generator.cuda()
        discriminator = discriminator.cuda()
        perceptualnet = perceptualnet.cuda()
    
    # ----------------------------------------
    #       Initialize training dataset
    # ----------------------------------------

    # Define the dataset
    trainset = train_dataset.InpaintDataset(opt)
    print('The overall number of images equals to %d' % len(trainset))

    # Define the dataloader
    dataloader = DataLoader(trainset, batch_size = opt.batch_size, shuffle = False, num_workers = opt.num_workers, pin_memory = True, drop_last=True)
    
    # ----------------------------------------
    #            Training
    # ----------------------------------------

    # Initialize start time
    prev_time = time.time()
    
    # Tensor type
    Tensor = torch.cuda.FloatTensor

    rend = Render(img_size=[opt.imgsize,opt.imgsize], batch_size=opt.batch_size, device=device) # [256,256]

    with open('results/log.txt', 'a') as log_file:
        log_file.write("\r\n#------------------------- new_start -------------------------#")
    # Training loop
    for epoch in range(opt.resume_epoch, opt.epochs):
        for batch_idx, (img, gt_img, mask, verts) in enumerate(dataloader):

            img = img.cuda()
            gt_img = gt_img.cuda()
            mask = mask.cuda()
            verts = verts.cuda()
            # set the same free form masks for each batch
            # mask = torch.empty(img.shape[0], 1, img.shape[2], img.shape[3]).cuda()
            # for i in range(opt.batch_size):
            #     mask[i] = torch.from_numpy(train_dataset.InpaintDataset.random_ff_mask(
            #                                     shape=(height[0], width[0])).astype(np.float32)).cuda()
            
            # LSGAN vectors
            valid = Tensor(np.ones((img.shape[0], 1, img.shape[2]//32, img.shape[3]//32)))
            fake = Tensor(np.zeros((img.shape[0], 1, img.shape[2]//32, img.shape[3]//32)))
            zero = Tensor(np.zeros((img.shape[0], 1, img.shape[2]//32, img.shape[3]//32)))

            ### Train Discriminator
            optimizer_d.zero_grad()

            # Generator output
            first_out, second_out = generator(img, mask)

            # forward propagation
            first_out_wholeimg = img * (1 - mask) + first_out * mask        # in range [0, 1]
            second_out_wholeimg = img * (1 - mask) + second_out * mask      # in range [0, 1]

            rend.load_texture(second_out_wholeimg)
            fake_renderd = rend.get_renderd_tensor(verts)[:,:,:,:3]
            fake_renderd = fake_renderd.permute(0, 3, 1, 2)

            rend.load_texture(gt_img)
            true_renderd = rend.get_renderd_tensor(verts)[:,:,:,:3]
            true_renderd = true_renderd.permute(0, 3, 1, 2)

            # # Fake samples
            fake_scalar = discriminator(fake_renderd.detach()) # , mask
            # # True samples
            true_scalar = discriminator(true_renderd)
            #
            # # Loss and optimize
            loss_fake = -torch.mean(torch.min(zero, -valid-fake_scalar))
            loss_true = -torch.mean(torch.min(zero, -valid+true_scalar))

            # Overall Loss and optimize
            loss_D = 0.5 * (loss_fake + loss_true)
            loss_D.backward()
            optimizer_d.step()

            ### Train Generator
            optimizer_g.zero_grad()

            # L1 Loss
            first_L1Loss = (first_out - gt_img).abs().mean()
            second_L1Loss = (second_out - gt_img).abs().mean()
            
            # GAN Loss
            fake_scalar = discriminator(fake_renderd) # second_out_wholeimg, mask
            GAN_Loss = -torch.mean(fake_scalar)
            #
            # # Get the deep semantic feature maps, and compute Perceptual Loss
            # img_featuremaps = perceptualnet(gt_img)                          # feature maps
            # second_out_featuremaps = perceptualnet(second_out)
            # second_PerceptualLoss = L1Loss(second_out_featuremaps, img_featuremaps)

            # Compute losses
            loss = opt.lambda_l1 * first_L1Loss + opt.lambda_l1 * second_L1Loss + opt.lambda_gan * GAN_Loss#+ \
                   #opt.lambda_perceptual * second_PerceptualLoss
            loss.backward()
            optimizer_g.step()

            # Determine approximate time left
            batches_done = epoch * len(dataloader) + batch_idx
            batches_left = opt.epochs * len(dataloader) - batches_done
            time_left = datetime.timedelta(seconds = batches_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            if batch_idx % 40 == 0:
                log = "\r[Epoch %d/%d] [Batch %d/%d] [first Mask L1 Loss: %.5f] [second Mask L1 Loss: %.5f] " \
                      "[G Loss: %.5f] [D Loss: %.5f] time_left: %s" % (
                (epoch + 1), opt.epochs, batch_idx, len(dataloader), first_L1Loss.item(), second_L1Loss.item(),
                GAN_Loss.item(), loss_D.item(), time_left)

                # log = "\r[Epoch %d/%d] [Batch %d/%d] [first Mask L1 Loss: %.5f] [second Mask L1 Loss: %.5f] " \
                #       "[D Loss: %.5f] [G Loss: %.5f] [Perceptual Loss: %.5f] time_left: %s" % (
                # (epoch + 1), opt.epochs, batch_idx, len(dataloader), first_L1Loss.item(), second_L1Loss.item(),
                # loss_D.item(), GAN_Loss.item(), second_PerceptualLoss.item(), time_left)

                with open('results/log.txt','a') as log_file:
                    log_file.write(log)
                print(log)

            
            masked_img = gt_img * (1 - mask) + mask
            mask = torch.cat((mask, mask, mask), 1)
            if (batch_idx + 1) % 40 == 0:
                img_list = [gt_img, mask, masked_img, first_out, second_out]
                name_list = ['gt', 'mask', 'masked_img', 'first_out', 'second_out']
                utils.save_sample_png(sample_folder = sample_folder, sample_name = 'epoch%d' % (epoch + 1), img_list = img_list, name_list = name_list, pixel_max_cnt = 255)

        # Learning rate decrease
        adjust_learning_rate(opt.lr_g, optimizer_g, (epoch + 1), opt)
        adjust_learning_rate(opt.lr_d, optimizer_d, (epoch + 1), opt)

        # Save the model
        save_model_generator(generator, (epoch + 1), opt)
        save_model_discriminator(discriminator, (epoch + 1), opt)

        # ### Sample data every epoch
        # if (epoch + 1) % 1 == 0:
        #     img_list = [img, mask, masked_img, first_out, second_out]
        #     name_list = ['gt', 'mask', 'masked_img', 'first_out', 'second_out']
        #     utils.save_sample_png(sample_folder = sample_folder, sample_name = 'epoch%d' % (epoch + 1), img_list = img_list, name_list = name_list, pixel_max_cnt = 255)
