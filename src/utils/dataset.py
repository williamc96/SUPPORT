import skimage.io as skio
import numpy as np
import torch
import zarr

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from src.utils.util import get_coordinate,get_coordinate_generator

import numpy as np
import random
from tqdm import tqdm
import math
import tifffile
import napari

def normalize(image):
    """
    Normalize the image to [mean/std]=[0/1]

    Arguments:
        image: image stack (Pytorch Tensor with dimension [T, X, Y])

    Returns:
        image: normalized image stack (Pytorch Tensor with dimension [T, X, Y])
        mean_image: mean of the image stack (np.float)
        std_image: standard deviation of the image stack (np.float)
    """
    mean_image = torch.mean(image)
    std_image = torch.std(image)

    image -= mean_image
    image /= std_image

    return image, mean_image, std_image

class FrameReader:
    def __init__(self, fileID, width=588, height=624, gap=1728,
                 dtype=np.uint8, maxFrames=24000, shuffle=True):
        self.fileID = fileID
        self.width = width
        self.height = height
        self.gap = gap
        self.dtype = dtype
        if self.fileID.lower().endswith(('.tif', '.tiff')):
            with tifffile.TiffFile(self.fileID) as tif:
                self.maxFrames = len(tif.pages)
                # update width/height from the first page
                h, w = tif.pages[0].asarray().shape
                self.height, self.width = h, w
        
        if maxFrames!=-1:
            self.maxFrames = maxFrames
        self.frame_size = self.width * self.height
        self.shuffle = shuffle
        # pointer for non‐shuffled reads
        self.pointer = -1 if shuffle else 0

    def isDone(self, numFrames, givePartial=True):
        if self.pointer == self.maxFrames + 1:
            return True
        if givePartial:
            return False
        return self.pointer + numFrames > self.maxFrames

    def getFrames(self, numFrames=50, givePartial=True):
        start = (random.randint(0, self.maxFrames - numFrames)
                    if self.shuffle else self.pointer)
        # If TIFF stack, load via skimage
        if self.fileID.lower().endswith(('.tif', '.tiff')):
            with tifffile.TiffFile(self.fileID) as tif:
                total = len(tif.pages)
                if total == 0:
                    return np.array([])
                if not givePartial and numFrames > total:
                    return np.array([])
                end = start + numFrames
                # read pages on‐demand to avoid full memory load
                images = [page.asarray() for page in tif.pages[start:end]]
                
                if not self.shuffle:
                    self.pointer += numFrames
                return np.stack(images)

        # else raw reader
        if self.isDone(numFrames, givePartial):
            return np.array([])

        offset = (self.frame_size + self.gap) * start

        images = []
        with open(self.fileID, 'rb') as file:
            file.seek(offset)
            for _ in range(numFrames):
                data = np.fromfile(file,
                                   dtype=self.dtype,
                                   count=self.frame_size)
                images.append(data.reshape(self.height, self.width))
                file.seek(self.gap, 1)
                
        if not self.shuffle:
            self.pointer += numFrames
        return np.array(images)

def random_transform(input, target, rng, is_rotate=True):
    """
    Randomly rotate/flip the image

    Arguments:
        input: input image stack (Pytorch Tensor with dimension [b, T, X, Y])
        target: targer image stack (Pytorch Tensor with dimension [b, T, X, Y]), can be None
        rng: numpy random number generator
    
    Returns:
        input: randomly rotated/flipped input image stack (Pytorch Tensor with dimension [b, T, X, Y])
        target: randomly rotated/flipped target image stack (Pytorch Tensor with dimension [b, T, X, Y])
    """
    rand_num = rng.integers(0, 4) # random number for rotation
    rand_num_2 = rng.integers(0, 2) # random number for flip

    if is_rotate:
        if rand_num == 1:
            input = torch.rot90(input, k=1, dims=(2, 3))
            if target is not None:
                target = torch.rot90(target, k=1, dims=(2, 3))
        elif rand_num == 2:
            input = torch.rot90(input, k=2, dims=(2, 3))
            if target is not None:
                target = torch.rot90(target, k=2, dims=(2, 3))
        elif rand_num == 3:
            input = torch.rot90(input, k=3, dims=(2, 3))
            if target is not None:
                target = torch.rot90(target, k=3, dims=(2, 3))
    
    if rand_num_2 == 1:
        input = torch.flip(input, dims=[2])
        if target is not None:
            target = torch.flip(target, dims=[2])

    return input, target





class DatasetSUPPORT(Dataset):
    def __init__(self, noisy_images, patch_size=[61, 128, 128], patch_interval=[10, 64, 64], load_to_memory=True,\
        transform=None, random_patch=True, random_patch_seed=0):
        """
        Arguments:
            noisy_images: list of noisy image stack ([Tensor with dimension [t, x, y]])
            patch_size: size of the patch ([int]), ([t, x, y])
            patch_interval: interval between each patch ([int]), ([t, x, y])
            load_to_memory: whether load data into memory or not (bool)
            transform: function of transformation (function)
            random_patch: sample patch in random or not (bool)
            random_patch_seed: seed for randomness (int)
            algorithm: the algorithm of use (str)
        """
        # check arguments
        if len(patch_size) != 3:
            raise Exception("length of patch_size must be 3")
        if len(patch_interval) != 3:
            raise Exception("length of patch_interval must be 3")      

        # initialize
        self.data_weight = []
        for noisy_image in noisy_images:
            if load_to_memory:
                self.data_weight.append(torch.numel(noisy_image))
            else:
                self.data_weight.append(np.prod(noisy_image.shape))

        self.patch_size = patch_size
        self.patch_interval = patch_interval
        self.transform = transform
        self.random_patch = random_patch
        self.patch_rng = np.random.default_rng(random_patch_seed)
        self.precomputed_indices = None
        self.load_to_memory = load_to_memory

        self.noisy_images = noisy_images
        self.mean_images = []
        self.std_images = []
        if load_to_memory:
            for idx, noisy_image in enumerate(tqdm(noisy_images)):
                noisy_image, mean_image, std_image = normalize(noisy_image)
                self.noisy_images[idx] = noisy_image
                self.mean_images.append(mean_image)
                self.std_images.append(std_image)
                self.mean_images = torch.tensor(self.mean_images)
                self.std_images = torch.tensor(self.std_images)

        # generate index
        self.indices_ds = []
        for noisy_image in self.noisy_images:
            indices = []
            tmp_size = noisy_image.shape
            if np.any(tmp_size < np.array(self.patch_size)):
                raise Exception("patch size is larger than data size")

            for k in range(3):
                z_range = list(range(0, tmp_size[k]-self.patch_size[k]+1, self.patch_interval[k]))
                if tmp_size[k] - self.patch_size[k] > z_range[-1]:
                    z_range.append(tmp_size[k]-self.patch_size[k])
                indices.append(z_range)
            self.indices_ds.append(indices)

    def precompute_indices(self):
        """
        Precompute random patch indices for each image using vectorized operations.
        This function accounts for images of different sizes by generating random
        indices within the valid range for each image.
        """
        precomputed_indices = []
        
        # Iterate over each image in the dataset
        for ds_idx, noisy_image in enumerate(self.noisy_images):
            # Get the shape of the image (T, H, W)
            shape = noisy_image.shape
            
            # Determine the number of patches available for this image.
            # Here, we use the precomputed indices list from __init__ (indices_ds)
            # which was generated based on patch_size and patch_interval.
            indices_lists = self.indices_ds[ds_idx]
            count_i = len(indices_lists[0]) * len(indices_lists[1]) * len(indices_lists[2])
            
            # Calculate the valid range for each dimension
            t_range = shape[0] - self.patch_size[0] + 1
            y_range = shape[1] - self.patch_size[1] + 1
            z_range = shape[2] - self.patch_size[2] + 1
            
            # Generate random indices in a vectorized way for the current image
            t_indices = self.patch_rng.integers(0, t_range, size=count_i)
            y_indices = self.patch_rng.integers(0, y_range, size=count_i)
            z_indices = self.patch_rng.integers(0, z_range, size=count_i)
            
            # Create a list of tuples (ds_idx, t_idx, y_idx, z_idx) for this image
            indices_for_image = [(ds_idx, int(t), int(y), int(z))
                                for t, y, z in zip(t_indices, y_indices, z_indices)]
            precomputed_indices.extend(indices_for_image)
        
        # Shuffle the complete list of precomputed indices to randomize order per epoch
        self.patch_rng.shuffle(precomputed_indices)
        self.precomputed_indices = precomputed_indices

    def __len__(self):
        total = 0
        for indices in self.indices_ds:
            total += len(indices[0]) * len(indices[1]) * len(indices[2])

        return total

    def __getitem__(self, i):
        # slicing
        if self.random_patch:
            ds_idx, t_idx, y_idx, z_idx = self.precomputed_indices[i]
        else:
            ds_idx = 0
            t_idx = self.indices_ds[ds_idx][0][i // (len(self.indices_ds[ds_idx][1]) * len(self.indices_ds[ds_idx][2]))]
            y_idx = self.indices_ds[ds_idx][1][(i % (len(self.indices_ds[ds_idx][1]) * len(self.indices_ds[ds_idx][2]))) // len(self.indices_ds[ds_idx][2])]
            z_idx = self.indices_ds[ds_idx][2][i % len(self.indices_ds[ds_idx][2])]

        # input dataset range
        t_range = slice(t_idx, t_idx + self.patch_size[0])
        y_range = slice(y_idx, y_idx + self.patch_size[1])
        z_range = slice(z_idx, z_idx + self.patch_size[2])
        
        if self.load_to_memory:
            noisy_image = self.noisy_images[ds_idx][t_range, y_range, z_range]
        else:
            noisy_image_avg = torch.tensor(self.noisy_images[ds_idx].attrs["mean"])
            noisy_image_std = torch.tensor(self.noisy_images[ds_idx].attrs["std"])
            noisy_image = self.noisy_images[ds_idx][t_range, y_range, z_range]
            noisy_image = torch.tensor(noisy_image, dtype=torch.float32)
            return noisy_image, torch.tensor([[t_idx, t_idx + self.patch_size[0]],\
                [y_idx, y_idx + self.patch_size[1]], [z_idx, z_idx + self.patch_size[2]]]), torch.tensor(ds_idx), noisy_image_avg, noisy_image_std
        
        return noisy_image, torch.tensor([[t_idx, t_idx + self.patch_size[0]],\
            [y_idx, y_idx + self.patch_size[1]], [z_idx, z_idx + self.patch_size[2]]]), torch.tensor(ds_idx)




class DatasetSUPPORT_test_stitch(Dataset):
    def __init__(self, noisy_image, patch_size=[61, 128, 128], patch_interval=[10, 64, 64], load_to_memory=True,\
        transform=None, random_patch=False, random_patch_seed=0):
        """
        Arguments:
            noisy_image: noisy image stack (Tensor with dimension [t, x, y])
            patch_size: size of the patch ([int]), ([t, x, y])
            patch_interval: interval between each patch ([int]), ([t, x, y])
            load_to_memory: whether load data into memory or not (bool)
            transform: function of transformation (function)
            random_patch: sample patch in random or not (bool)
            random_patch_seed: seed for randomness (int)
        """
        # check arguments
        if len(patch_size) != 3:
            raise Exception("length of patch_size must be 3")
        if len(patch_interval) != 3:
            raise Exception("length of patch_interval must be 3")

        self.patch_size = patch_size
        self.patch_interval = patch_interval
        self.transform = transform
        self.random_patch = random_patch
        self.patch_rng = np.random.default_rng(random_patch_seed)
        self.noisy_image = noisy_image

        # generate index
        self.indices = []
        tmp_size = self.noisy_image.size()
        if np.any(tmp_size < np.array(self.patch_size)):
            raise Exception("patch size is larger than data size")

        self.indices = get_coordinate(tmp_size, patch_size, patch_interval)            

    def __len__(self):
        return len(self.indices) # len(self.indices[0]) * len(self.indices[1]) * len(self.indices[2])

    def __getitem__(self, i):
        # slicing
        if self.random_patch:
            idx = self.patch_rng.integers(0, len(self.indices) - 1)
        else:
            idx = i
        single_coordinate = self.indices[idx]
        
        # input dataset range
        init_h = single_coordinate['init_h']
        end_h = single_coordinate['end_h']
        init_w = single_coordinate['init_w']
        end_w = single_coordinate['end_w']
        init_s = single_coordinate['init_s']
        end_s = single_coordinate['end_s']


        # for stitching dataset range
        noisy_image = self.noisy_image[init_s:end_s,init_h:end_h,init_w:end_w]
        
        noisy_image, mean_image, std_image = normalize(noisy_image)
        # print(single_coordinate)
        # print(noisy_image.shape)
        # transform
        if self.transform:
            rand_i = self.patch_rng.integers(0, self.transform.n_masks)
            rand_t = self.patch_rng.integers(0, 2)
            noisy_image = self.transform.mask(noisy_image, rand_i, rand_t)

        return noisy_image, torch.empty(1), single_coordinate, mean_image, std_image

class DatasetSUPPORT_incremental_load(Dataset):
    def __init__(self, reader, patch_size=[61, 128, 128], patch_interval=[10, 64, 64],\
        transform=None, batch_size = -1, maxItems=None):
        if (batch_size==-1):
            batch_size = patch_size[0]*4
        if len(patch_size) != 3:
            raise Exception("length of patch_size must be 3")
        if len(patch_interval) != 3:
            raise Exception("length of patch_interval must be 3")
        self.patch_size = patch_size
        self.patch_interval = patch_interval
        self.transform = transform
        self.reader = reader
        self.batch_size = batch_size
        self.canBeFirstInBatch = True
        
        # load and convert uint16 frames to float32
        frames = self.reader.getFrames(self.batch_size).astype(np.float32)
        self.noisy_image = torch.from_numpy(frames)
        print("init: " + str(self.noisy_image.shape))
        self.indices = []
        tmp_size = self.noisy_image.size()
        if np.any(tmp_size < np.array(self.patch_size)):
            raise Exception("patch size is larger than data size")
        tmp_size = list(tmp_size)  # convert the torch.Size to a list
        tmp_size[0] = reader.maxFrames  # update the element
        tmp_size = torch.Size(tmp_size)  # convert the list back to torch.Size
        self.output_size = tmp_size
        self.coordinate_gen = get_coordinate_generator(tmp_size, patch_size, patch_interval)


        ptr = self.reader.pointer
        framesForBaseline = self.reader.getFrames(max(self.batch_size,500)).astype(np.float32)
        self.reader.pointer = ptr # reset the pointer to the original position

        whole_s, whole_h, whole_w = reader.maxFrames, reader.width, reader.height
        img_s, img_h, img_w = patch_size
        gap_s, gap_h, gap_w = patch_interval
        num_w = math.ceil((whole_w-img_w+gap_w)/gap_w)
        num_h = math.ceil((whole_h-img_h+gap_h)/gap_h)
        num_s = math.ceil((whole_s-img_s+gap_s)/gap_s)
        self.length = num_w*num_h*num_s
        self.maxItems = maxItems

    def __len__(self):
        if self.maxItems is not None:
            return min(self.length, self.maxItems)
        return self.length

    def numExtra(self):
        return math.ceil((self.patch_size[0]-1)/2)

    def __getitem__(self, i):

        single_coordinate = next(self.coordinate_gen)
        init_h = single_coordinate['init_h']
        end_h = single_coordinate['end_h']
        init_w = single_coordinate['init_w']
        end_w = single_coordinate['end_w']
        init_s = single_coordinate['init_s']
        end_s = single_coordinate['end_s']

        
        # print("frames shape: " + str(self.noisy_image.shape))

        # print("before init: " + str(init_s) + " end: " + str(end_s))
        difference = end_s - init_s
        if self.reader.shuffle:
            init_s = init_s % (self.batch_size - difference)
            if init_s == 0 and self.canBeFirstInBatch:
                self.append_frames()
                init_s = 0
                self.canBeFirstInBatch = False
            else:
                self.canBeFirstInBatch = init_s != 0
            end_s = (init_s + difference)
        else:
            if end_s>self.reader.pointer:
                self.append_frames()
            frameZeroIndex = self.reader.pointer - len(self.noisy_image)
            init_s = init_s-frameZeroIndex
            end_s = end_s-frameZeroIndex
            
        # print("after init: " + str(init_s) + " end: " + str(end_s) + " self.reader.pointer: " + str(self.reader.pointer) + " len(self.noisy_image): " + str(len(self.noisy_image)))
        
        noisy_image = self.noisy_image[init_s:end_s,init_h:end_h,init_w:end_w]
        
        noisy_image = torch.from_numpy(np.array(noisy_image)).type(torch.FloatTensor)
        noisy_image, mean_image, std_image = normalize(noisy_image)
        # print(noisy_image.shape)
        # print(single_coordinate)
        if self.transform:
            rand_i = self.patch_rng.integers(0, self.transform.n_masks)
            rand_t = self.patch_rng.integers(0, 2)
            noisy_image = self.transform.mask(noisy_image, rand_i, rand_t)
        return noisy_image, -15, single_coordinate, mean_image, std_image

    def append_frames(self):
        frames = self.reader.getFrames(self.batch_size).astype(np.float32)
        newFrames   = torch.from_numpy(frames).type(torch.FloatTensor)

        # print("NEW FRAMES: " + str(newFrames.shape))
        # Keep the last `self.numExtra()` frames of the old images
        if self.reader.shuffle:
            self.noisy_image = newFrames
        else:
            self.noisy_image = torch.cat((self.noisy_image[-self.numExtra()*2:], newFrames), dim=0)



def gen_train_dataloader(patch_size, patch_interval, batch_size, noisy_data_list, totalFrames=10000,numConsecFrames=200):
    """
    Generate dataloader for training

    Arguments:
        patch_size: opt.patch_size
        patch_interval: opt.patch_interval
        noisy_data_list: opt.noisy_data
    
    Returns:
        dataloader_train
    """
    noisy_images_train = []

    numFiles = len(noisy_data_list)
    print("=============== loading data ===============")
    for i,noisy_data in enumerate(noisy_data_list):
        print("file " + str(i+1) + " of " + str(numFiles))
        if noisy_data.endswith('.raw'):
            frameReader = FrameReader(noisy_data)
            for _ in tqdm(range(int(totalFrames/numFiles/numConsecFrames)), desc="Loading file {}".format(i+1), ncols=70):
                frames = frameReader.getFrames(numConsecFrames)
                noisy_image = torch.from_numpy(frames.astype(np.float32)).type(torch.FloatTensor)
                T, _, _ = noisy_image.shape
                noisy_images_train.append(noisy_image)
        else:
            noisy_image = torch.from_numpy(skio.imread(noisy_data).astype(np.float32)).type(torch.FloatTensor)
            T, _, _ = noisy_image.shape
            noisy_images_train.append(noisy_image)


    dataset_train = DatasetSUPPORT(noisy_images_train, patch_size=patch_size,\
        patch_interval=patch_interval, transform=None, random_patch=True)
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    print("=============== done loading ===============")
    return dataloader_train

def gen_test_dataloader(patch_size, patch_interval, batch_size, noisy_data_list, totalFrames=10000,numConsecFrames=200):
    """
    Generate dataloader for training

    Arguments:
        patch_size: opt.patch_size
        patch_interval: opt.patch_interval
        noisy_data_list: opt.noisy_data
    
    Returns:
        dataloader_train
    """
    noisy_images_train = []

    numFiles = len(noisy_data_list)
    print("=============== loading data ===============")
    for i,noisy_data in enumerate(noisy_data_list):
        print("file " + str(i+1) + " of " + str(numFiles))
        if noisy_data.endswith('.raw'):
            frameReader = FrameReader(noisy_data)
            for _ in tqdm(range(int(totalFrames/numFiles/numConsecFrames)), desc="Loading file {}".format(i+1), ncols=70):
                frames = frameReader.getFrames(numConsecFrames)
                noisy_image = torch.from_numpy(frames.astype(np.float32)).type(torch.FloatTensor)
                T, _, _ = noisy_image.shape
                noisy_images_train.append(noisy_image)
        else:
            noisy_image = torch.from_numpy(skio.imread(noisy_data).astype(np.float32)).type(torch.FloatTensor)
            T, _, _ = noisy_image.shape
            noisy_images_train.append(noisy_image)


    dataset_train = DatasetSUPPORT(noisy_images_train, patch_size=patch_size,\
        patch_interval=patch_interval, transform=None, random_patch=True)
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    print("=============== done loading ===============")
    return dataloader_train