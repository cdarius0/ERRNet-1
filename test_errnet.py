from os.path import join, basename
from options.errnet.train_options import TrainOptions
from engine import Engine
from data.image_folder import read_fns
from data.transforms import __scale_width
import torch.backends.cudnn as cudnn
import data.reflect_dataset as datasets
import util.util as util
import datetime
opt = TrainOptions().parse()
import os
import argparse

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"  # specify which GPU(s) to be used

# there is still work do be done regarding training and testing on different gpus
# even when i use --gpu_ids the first gpu(0) is still used partialy. Check self.model in engine.py. Check torch.device
# Check self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


opt.isTrain = False
cudnn.benchmark = True
opt.no_log = True
opt.display_id = 0
opt.verbose = False
opt.resume = True  # means that test will use a set of weights specified as resume_epoch
# opt.resume_epoch = 'errnet_latest.pt'
# opt.hyper = True
engine = Engine(opt)

# For barney images / semne rutiere
# datadir = '/media/uic36713/Ext2TB/darius/imagini_barney_bariera_432x224'
# result_dir = './results/Barney_images_after_change_blurr_testing'

now = (str(datetime.datetime.now().replace(second=0, microsecond=0))).replace(':', '_').replace(' ', '.')[:-3]
# '2019-12-11.14_38'

# datadir = '/data/Datasets/darius_stray_light/darius_test_dataset_real_225_MFC520/'
datadir = '/data/Datasets/darius_stray_light/darius_test_5_images'
result_dir = opt.checkpoints_dir+'/'+opt.name+'/results_inference/epoch_'+str(engine.epoch)+'_'+now+'/'

test_dataset_real = datasets.CEILTestDataset(datadir, enable_transforms=False)
test_dataloader = datasets.DataLoader(
    test_dataset_real, batch_size=1, shuffle=False,
    num_workers=opt.nThreads, pin_memory=True)

# When using generated reflexion images 150

# datadir = '/media/uic36713/Ext2TB/darius/pentru_demo_8.11/modified_data/'
# datadir_syn = join(datadir, 'overlay')
# datadir_real = join(datadir, 'transmission_layer')
# result_dir = './results/test_on_150_change_blurr_refl_2_default_kernel/'
#
# test_dataset = datasets.CEILTestDataset_darius(
#     datadir=datadir_real, datadir_syn=datadir_syn, fns=read_fns('z_darius_listed_files/for_test_gt.txt'),
#     fns_syn=read_fns('z_darius_listed_files/for_test_reflection.txt'),
#     size=150,
#     enable_transforms=False,
#     low_sigma=opt.low_sigma, high_sigma=opt.high_sigma,
#     low_gamma=opt.low_gamma, high_gamma=opt.high_gamma, opacity=0.75)
#
# # test_dataset_real = datasets.CEILTestDataset_darius(datadir_real, read_fns('z_darius_listed_files/for_test_gt.txt'),
# #                                                      enable_transforms=True)
#
# test_dataset_fusion = datasets.FusionDataset([test_dataset], [1.0])
# test_dataloader = datasets.DataLoader(
#     test_dataset_fusion, batch_size=1, shuffle=not opt.serial_batches,
#     num_workers=opt.nThreads, pin_memory=True)


"""Main Loop copied from below"""
res = engine.test(test_dataloader, savedir=result_dir)



# Define evaluation/test dataset

# eval_dataset_ceilnet = datasets.CEILTestDataset(join(datadir, 'testdata_CEILNET_table2'))
# eval_dataset_sir2 = datasets.CEILTestDataset(join(datadir, 'sir2_withgt'))


# eval_dataset_real = datasets.CEILTestDataset(
#     datadir,
#     # fns=read_fns('for_test_blended.txt'),
#     fns=read_fns('abczzz.txt'),
#     # fnstrans_layer = read_fns('real_test_trans_layer.txt'),
#     size=150)


# eval_dataset_real = datasets.CEILDataset(
#     datadir, read_fns('abczzz.txt'), size=opt.max_dataset_size, enable_transforms=True,
#     low_sigma=opt.low_sigma, high_sigma=opt.high_sigma,
#     low_gamma=opt.low_gamma, high_gamma=opt.high_gamma)


# eval_dataset_postcard = datasets.CEILTestDataset(join(datadir, 'postcard'))
# eval_dataset_solidobject = datasets.CEILTestDataset(join(datadir, 'solidobject'))

# test_dataset_internet = datasets.RealDataset(join(datadir, 'internet'))
# test_dataset_unaligned300 = datasets.RealDataset(join(datadir, 'refined_unaligned_data/unaligned300/blended'))
# test_dataset_unaligned150 = datasets.RealDataset(join(datadir, 'refined_unaligned_data/unaligned150/blended'))
# test_dataset_unaligned_dynamic = datasets.RealDataset(join(datadir, 'refined_unaligned_data/unaligned_dynamic/blended'))
# test_dataset_sir2 = datasets.RealDataset(join(datadir, 'sir2_wogt/blended'))


# eval_dataloader_ceilnet = datasets.DataLoader(
#     eval_dataset_ceilnet, batch_size=1, shuffle=False,
#     num_workers=opt.nThreads, pin_memory=True)





# eval_dataloader_real = datasets.DataLoader(
#     train_dataset_fusion, batch_size=1, shuffle=False,
#     num_workers=opt.nThreads, pin_memory=True)






# eval_dataloader_sir2 = datasets.DataLoader(
#     eval_dataset_sir2, batch_size=1, shuffle=False,
#     num_workers=opt.nThreads, pin_memory=True)
#
# eval_dataloader_solidobject = datasets.DataLoader(
#     eval_dataset_solidobject, batch_size=1, shuffle=False,
#     num_workers=opt.nThreads, pin_memory=True)
#
# eval_dataloader_postcard = datasets.DataLoader(
#     eval_dataset_postcard, batch_size=1, shuffle=False,
#     num_workers=opt.nThreads, pin_memory=True)

# test_dataloader_internet = datasets.DataLoader(
#     test_dataset_internet, batch_size=1, shuffle=False,
#     num_workers=opt.nThreads, pin_memory=True)

# test_dataloader_sir2 = datasets.DataLoader(
#     test_dataset_sir2, batch_size=1, shuffle=False,
#     num_workers=opt.nThreads, pin_memory=True)

# test_dataloader_unaligned300 = datasets.DataLoader(
#     test_dataset_unaligned300, batch_size=1, shuffle=False,
#     num_workers=opt.nThreads, pin_memory=True)

# test_dataloader_unaligned150 = datasets.DataLoader(
#     test_dataset_unaligned150, batch_size=1, shuffle=False,
#     num_workers=opt.nThreads, pin_memory=True)

# test_dataloader_unaligned_dynamic = datasets.DataLoader(
#     test_dataset_unaligned_dynamic, batch_size=1, shuffle=False,
#     num_workers=opt.nThreads, pin_memory=True)


# """Main Loop"""
#
# engine = Engine(opt)
#
# # evaluate on synthetic test data from CEILNet
# # res = engine.eval(eval_dataloader_real, savedir=result_dir)
#
# # evaluate on four real-world benchmarks
# # res = engine.eval(eval_dataloader_real, dataset_name='testdata_real')
#
# # res = engine.eval(eval_dataloader_real, dataset_name='testdata_real', savedir=join(result_dir, 'real20'))
# # res = engine.eval(eval_dataloader_postcard, dataset_name='testdata_postcard', savedir=join(result_dir, 'postcard'))
# # res = engine.eval(eval_dataloader_sir2, dataset_name='testdata_sir2', savedir=join(result_dir, 'sir2_withgt'))
# # res = engine.eval(eval_dataloader_solidobject, dataset_name='testdata_solidobject', savedir=join(result_dir, 'solidobject'))
#
# res = engine.test(test_dataloader, savedir=result_dir)
#
# # test on our collected unaligned data or internet images
# # res = engine.test(test_dataloader_internet, savedir=join(result_dir, 'internet'))
# # res = engine.test(test_dataloader_unaligned300, savedir=join(result_dir, 'unaligned300'))
# # res = engine.test(test_dataloader_unaligned150, savedir=join(result_dir, 'unaligned150'))
# # res = engine.test(test_dataloader_unaligned_dynamic, savedir=join(result_dir, 'unaligned_dynamic'))
# # res = engine.test(test_dataloader_sir2, savedir=join(result_dir, 'sir2_wogt'))
