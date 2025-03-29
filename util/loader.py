import random 
import os
import numpy as np
from tqdm import tqdm 
import torch
import torchvision.datasets as datasets
import glob 
import pickle
class ImageFolderWithFilename(datasets.ImageFolder):
    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target, filename).
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        filename = path.split(os.path.sep)[-2:]
        filename = os.path.join(*filename)
        return sample, target, filename

class CachedFolder(datasets.DatasetFolder):
    def __init__(
            self,
            root: str,
    ):
        super().__init__(
            root,
            loader=None,
            extensions=(".npz",),
        )
        print("yes")
        '''
        filter_samples=[]
        for i in tqdm(range(len(self.samples))):
            path, target = self.samples[i]
            indices_path=path.replace('cache_dcae','cache_vaekl_indices_0210').replace('npz','pth')
            if not os.path.exists(indices_path):
                continue
            filter_samples.append(self.samples[i])
        sup_paths=glob.glob('/nfs-134/zhangguiwei/cache_dcae/val/*.npz')
        sup_paths+=glob.glob('/nfs-134/zhangguiwei/cache_dcae/test/*.npz')
        for path in sup_paths:
            target=0
            indices_path=path.replace('cache_dcae','cache_vaekl_indices_0210').replace('npz','pth')
            if not os.path.exists(indices_path):
                continue
            filter_samples.append((path,target))
        self.samples=filter_samples
        '''
       # self.samples+=pickle.load(open('lmdb_filter_paths.pkl','rb'))
      #  self.samples=pickle.load(open('dcae_1024_cache_paths.pkl','rb'))
        self.samples=pickle.load(open('dcae_1024_cache_paths.pkl','rb'))
    
    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (moments, target).
        """
        path = self.samples[index]
        data = np.load(path)
        if torch.rand(1) < 0.5:  # randomly hflip
            moments = data['moments']
        else:
            moments = data['moments_flip']
        return moments
