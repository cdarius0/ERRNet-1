import os
import datetime
from shutil import copyfile
import argparse

exp_name = 'another_one_kernel[1,2]_new'
experiment_location = './checkpoints/'+exp_name+'/'

parser = argparse.ArgumentParser(description='Add params for inference over many epochs')
parser.add_argument('--exp_name', default=None,
                    help='Add the name of the experiment to the test_errnet.py called inside')

args = parser.parse_args()
for epoch in os.listdir(experiment_location):
    if '.pt' in epoch:
        epoch = epoch[:-3]
        os.system("python test_errnet.py --hyper --name {exp_name} -re {epoch}".format(exp_name=args.exp_name,
                                                                                       epoch=epoch))
inference_results_dir = '/data/Models/ERRNet_last/checkpoints/'+exp_name+'/results_inference/'
now = (str(datetime.datetime.now().replace(second=0, microsecond=0))).replace(':', '_').replace(' ', '.')[:-3]
combined_results_inference_dir = '/data/Models/ERRNet_last/checkpoints/'+exp_name+'/inference_results_COMPARE_EPOCHS/'
if not os.path.exists(combined_results_inference_dir):
    os.mkdir(combined_results_inference_dir)
combined_results_inference_dir += now+'/'
if not os.path.exists(combined_results_inference_dir):
    os.mkdir(combined_results_inference_dir)
for epoch_results in os.listdir(inference_results_dir):
    inference_images = os.listdir(inference_results_dir+epoch_results+'/')
    copyfile(inference_results_dir+epoch_results+'/'+inference_images[1]+'/output.bmp', combined_results_inference_dir+epoch_results+'.bmp')