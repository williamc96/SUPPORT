import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import numpy as np
import torch
import skimage.io as skio

from tqdm import tqdm
from src.utils.dataset import DatasetSUPPORT_test_stitch,FrameReader,DatasetSUPPORT_incremental_load
from model.SUPPORT import SUPPORT
from src.utils.util import parse_arguments

import h5py

def validate(test_dataloader, model, output_file):
    """
    Validate a model with a test data
    
    Arguments:
        test_dataloader: (Pytorch DataLoader)
            Should be DatasetFRECTAL_test_stitch!
        model: (Pytorch nn.Module)

    Returns:
        denoised_stack: denoised image stack (Numpy array with dimension [T, X, Y])
    """
    # get the shape of the noisy image stack for creating the HDF5 dataset
    stack_shape = test_dataloader.dataset.output_size
    
    with h5py.File(output_file, 'w') as hdf5_file:
        # create an HDF5 dataset to store the denoised stack
        denoised_stack = hdf5_file.create_dataset("denoised_stack", stack_shape, dtype=np.float32)
        
        with torch.no_grad():
            model.eval()
            # stitching denoised stack
            for _, (noisy_image, _, single_coordinate) in enumerate(tqdm(test_dataloader, desc="validate")):
                noisy_image = noisy_image.cuda() #[b, z, y, x]
                noisy_image_denoised = model(noisy_image)
                T = noisy_image.size(1)
                for bi in range(noisy_image.size(0)): 
                    stack_start_w = int(single_coordinate['stack_start_w'][bi])
                    stack_end_w = int(single_coordinate['stack_end_w'][bi])
                    patch_start_w = int(single_coordinate['patch_start_w'][bi])
                    patch_end_w = int(single_coordinate['patch_end_w'][bi])

                    stack_start_h = int(single_coordinate['stack_start_h'][bi])
                    stack_end_h = int(single_coordinate['stack_end_h'][bi])
                    patch_start_h = int(single_coordinate['patch_start_h'][bi])
                    patch_end_h = int(single_coordinate['patch_end_h'][bi])

                    stack_start_s = int(single_coordinate['init_s'][bi])
                    
                    denoised_stack[stack_start_s+(T//2), stack_start_h:stack_end_h, stack_start_w:stack_end_w] \
                        = noisy_image_denoised[bi].squeeze()[patch_start_h:patch_end_h, patch_start_w:patch_end_w].cpu().numpy()

        # change nan values to 0 and denormalize
        denoised_stack[...] = denoised_stack[...] * test_dataloader.dataset.std_image.numpy() + test_dataloader.dataset.mean_image.numpy()

# def validate(test_dataloader, model, output_file):
#     """
#     Validate a model with a test data
    
#     Arguments:
#         test_dataloader: (Pytorch DataLoader)
#             Should be DatasetFRECTAL_test_stitch!
#         model: (Pytorch nn.Module)

#     Returns:
#         denoised_stack: denoised image stack (Numpy array with dimension [T, X, Y])
#     """
#     with torch.no_grad():
#         model.eval()
#         # initialize denoised stack to NaN array.
#         denoised_stack = np.zeros(test_dataloader.dataset.noisy_image.shape, dtype=np.float32)
        
#         # stitching denoised stack
#         # insert the results if the stack value was NaN
#         # or, half of the output volume
#         for _, (noisy_image, _, single_coordinate) in enumerate(tqdm(test_dataloader, desc="validate")):
#             noisy_image = noisy_image.cuda() #[b, z, y, x]
#             noisy_image_denoised = model(noisy_image)
#             T = noisy_image.size(1)
#             for bi in range(noisy_image.size(0)): 
#                 stack_start_w = int(single_coordinate['stack_start_w'][bi])
#                 stack_end_w = int(single_coordinate['stack_end_w'][bi])
#                 patch_start_w = int(single_coordinate['patch_start_w'][bi])
#                 patch_end_w = int(single_coordinate['patch_end_w'][bi])

#                 stack_start_h = int(single_coordinate['stack_start_h'][bi])
#                 stack_end_h = int(single_coordinate['stack_end_h'][bi])
#                 patch_start_h = int(single_coordinate['patch_start_h'][bi])
#                 patch_end_h = int(single_coordinate['patch_end_h'][bi])

#                 stack_start_s = int(single_coordinate['init_s'][bi])
                
#                 denoised_stack[stack_start_s+(T//2), stack_start_h:stack_end_h, stack_start_w:stack_end_w] \
#                     = noisy_image_denoised[bi].squeeze()[patch_start_h:patch_end_h, patch_start_w:patch_end_w].cpu()

#         # change nan values to 0 and denormalize
#         denoised_stack = denoised_stack * test_dataloader.dataset.std_image.numpy() + test_dataloader.dataset.mean_image.numpy()

#         return denoised_stack


# if __name__ == '__main__':
#     opt = parse_arguments()
    
#     print(opt)
#     ########## Change it with your data ##############
#     data_file = opt.noisy_data[0]
#     model_file = opt.model # "./results/saved_models/mytest/model_0.pth"
#     output_file = "./results/denoised_0.tif"
#     patch_size = [61, 64, 64]
#     patch_interval = [1, 32, 32]
#     batch_size = 16    # lower it if memory exceeds.
#     ##################################################

#     model = SUPPORT(in_channels=opt.input_frames, mid_channels=opt.unet_channels, depth=opt.depth,\
#          blind_conv_channels=opt.blind_conv_channels, one_by_one_channels=opt.one_by_one_channels,\
#                 last_layer_channels=opt.last_layer_channels, bs_size=opt.bs_size, bp=opt.bp).cuda()
#     # model = SUPPORT(in_channels=61, mid_channels=opt.unet_channels, depth=opt.depth,\
#     #         blind_conv_channels=64, one_by_one_channels=[32, 16], last_layer_channels=[64, 32, 16], bs_size=bs_size).cuda()

#     model.load_state_dict(torch.load(model_file))

#     demoFile = torch.from_numpy(skio.imread(data_file).astype(np.float32)).type(torch.FloatTensor)
#     demoFile = demoFile[:, :, :]

#     testset = DatasetSUPPORT_test_stitch(demoFile, patch_size=patch_size,\
#         patch_interval=patch_interval)
#     testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size)
#     denoised_stack = validate(testloader, model)

#     print(denoised_stack.shape)
#     skio.imsave(output_file, denoised_stack[(model.in_channels-1)//2:-(model.in_channels-1)//2, : , :], metadata={'axes': 'TYX'})


if __name__ == '__main__':
    opt = parse_arguments()
    
    print(opt)
    ########## Change it with your data ##############
    data_file = opt.noisy_data[0]
    model_file = opt.model # "./results/saved_models/mytest/model_0.pth"
    output_file = "./results/denoised_0.tif"
    patch_size = [61, 64, 64]
    patch_interval = [1, 32, 32]
    batch_size = 16    # lower it if memory exceeds.
    ##################################################

    model = SUPPORT(in_channels=opt.input_frames, mid_channels=opt.unet_channels, depth=opt.depth,\
         blind_conv_channels=opt.blind_conv_channels, one_by_one_channels=opt.one_by_one_channels,\
                last_layer_channels=opt.last_layer_channels, bs_size=opt.bs_size, bp=opt.bp).cuda()
    # model = SUPPORT(in_channels=61, mid_channels=opt.unet_channels, depth=opt.depth,\
    #         blind_conv_channels=64, one_by_one_channels=[32, 16], last_layer_channels=[64, 32, 16], bs_size=bs_size).cuda()

    model.load_state_dict(torch.load(model_file))

    reader = FrameReader(data_file)
    # demoFile = torch.from_numpy(skio.imread(data_file).astype(np.float32)).type(torch.FloatTensor)
    # demoFile = demoFile[:, :, :]

    testset = DatasetSUPPORT_incremental_load(reader, patch_size=patch_size,\
        patch_interval=patch_interval)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size)
    denoised_stack = validate(testloader, model, output_file)

    # print(denoised_stack.shape)
    # skio.imsave(output_file, denoised_stack[(model.in_channels-1)//2:-(model.in_channels-1)//2, : , :], metadata={'axes': 'TYX'})

