from os.path import join
from options.errnet.train_options import TrainOptions
from engine import Engine
from data.image_folder import read_fns
import torch.backends.cudnn as cudnn
import data.reflect_dataset as datasets
import util.util as util
import data
import os

# TO DO: Maybe smaller lr because it seems to have some good results at epoch x and not marginaly bad at epoch x-1
# and x-2 but really bad. So change lr and change lr epoch to lower values
# gpu 0 in use even when training on another gpu
# maybe change gan type in train_options

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # specify which GPU(s) to be used
opt = TrainOptions().parse()

cudnn.benchmark = True
opt.batchSize = 1
opt.display_freq = 5000
opt.resume_epoch = 'errnet_latest'  # no need to add .pt
opt.resume = True

if opt.debug:
    opt.display_id = 1
    opt.display_freq = 20
    opt.print_freq = 20
    opt.nEpochs = 40
    opt.max_dataset_size = 100
    opt.no_log = False
    opt.nThreads = 0
    opt.decay_iter = 0
    opt.serial_batches = True
    opt.no_flip = True


#train on 93570
datadir_syn = 'D:/Google Drive Autonom/Datasets/SIRR/voc_reshaped_224x224/'
file_with_names_of_images = 'VOC17k_train.txt'

engine = Engine(opt)
# file_with_names_of_images = 'VOC17k_train.txt'
# maybe give them as params in command line and make a sh file for each training
train_dataset = datasets.CEILDataset(
    # datadir_syn, read_fns('VOC17k_train.txt'),
    datadir_syn, name_of_experiment=opt.name,
    fns=read_fns(file_with_names_of_images),
    size=opt.max_dataset_size, enable_transforms=True,
    low_sigma=0.2, high_sigma=2.0,  # low_sigma=2, high_sigma=5,
    low_gamma=1.3, high_gamma=1.3,  # you probably shouldn't change gamma from 1.3
    min_opacity=0.6, max_opacity=1.0,
    min_back_opacity=0.6, max_back_opacity=1.0,
    kernel_sizes=[1, 2], current_epoch=engine.epoch,  # vertical_flip_percentage=0.5,
    batch_size=opt.batchSize, display_freq=opt.display_freq, file_with_names_of_images=file_with_names_of_images,
    vertical_flip='Custom set in transforms.py depending on kernel.'
                  'Kernel 1 -> 1.0 vflip, Kernel 2 -> 0.0 vflip Kernel 3 = 0.5 vflip')
# check reflection_synthesis_parameters_options and opt.txt
train_dataloader_fusion = datasets.DataLoader(
    train_dataset, batch_size=opt.batchSize, shuffle=not opt.serial_batches,
    num_workers=opt.nThreads, pin_memory=True)

"""Main Loop"""


def set_learning_rate(lr):
    for optimizer in engine.model.optimizers:
        print('[i] set learning rate to {}'.format(lr))
        util.set_opt_param(optimizer, 'lr', lr)


# if opt.resume:
#     res = engine.eval(eval_dataloader_ceilnet, dataset_name='testdata_table2')

# define training strategy
max_epochs = 60
call_test_beginning_with_epoch = 100
add_gan_loss_at_epoch = 10
change_lr_at_epoch = [   0,    15,   25,   30,   40]
# change_lr_to =       [1e-4, 5e-5, 1e-5, 5e-5, 1e-5]
change_lr_to =       [1e-4, 5e-5, 1e-5, 5e-6, 1e-5]
engine.model.opt.lambda_gan = 0
set_learning_rate(change_lr_to[0])
# ratio_synthetic_to_real_training_images = [0.5, 0.5]
loss_lambda_gan = 0.01


file_name = './checkpoints/' + opt.name + '/reflection_synthesis_parameters_options'
with open(file_name, 'a+') as syntetical_reflection_opt:  # `a+` create file if it does not exist and append to it
    syntetical_reflection_opt.write("Max epochs:{max_epochs}\n"
                                    "Adding {loss_lambda_gan} gan loss at epoch {add_gan_loss_at_epoch}\n"
                                    .format(max_epochs=max_epochs, loss_lambda_gan=loss_lambda_gan,
                                            add_gan_loss_at_epoch=add_gan_loss_at_epoch))
    for epoch, learning_rate in zip(change_lr_at_epoch, change_lr_to):
        syntetical_reflection_opt.write("At epoch {epoch} we change the learning rate to {learning_rate}\n"
                                        .format(epoch=epoch, learning_rate=learning_rate))
    syntetical_reflection_opt.write("\n\n")

while engine.epoch < max_epochs:
    # while engine.epoch < opt.nEpochs: # 60 epochs
    # if engine.epoch >= 20:#max epoch 60 so maybe change accordingly to the nb of total epochs until overfit
    # if engine.epoch >= call_test_beginning_with_epoch:
    #     os.system("python test_errnet.py --hyper --name {exp_name} -re {epoch}".format(exp_name=opt.name,
    #                                                                                    epoch=engine.epoch))
    if engine.epoch >= add_gan_loss_at_epoch:
        engine.model.opt.lambda_gan = loss_lambda_gan
    # if engine.epoch >= 30:
    #     set_learning_rate(5e-5)
    # if engine.epoch >= 40:
    #     set_learning_rate(1e-5)
    # if engine.epoch >= 45:
    #     # ratio = ratio_synthetic_to_real_training_images
    #     # print('[i] adjust fusion ratio to {}'.format(ratio))
    #     # train_dataset_fusion.fusion_ratios = ratio
    #     set_learning_rate(5e-5)
    # if engine.epoch >= 50:
    #     set_learning_rate(1e-5)
    if engine.epoch >= change_lr_at_epoch[1]:
        set_learning_rate(change_lr_to[1])
    if engine.epoch >= change_lr_at_epoch[2]:
        set_learning_rate(change_lr_to[2])
    if engine.epoch >= change_lr_at_epoch[3]:
        # ratio = [0.5, 0.5]
        # print('[i] adjust fusion ratio to {}'.format(ratio))
        # train_dataset_fusion.fusion_ratios = ratio
        set_learning_rate(change_lr_to[3])
    if engine.epoch >= change_lr_at_epoch[4]:
        set_learning_rate(change_lr_to[4])
    engine.train(train_dataloader_fusion)

    # if engine.epoch % 5 == 0:
    #     engine.eval(eval_dataloader_ceilnet, dataset_name='testdata_table2')
    #     engine.eval(eval_dataloader_real, dataset_name='testdata_real20')

