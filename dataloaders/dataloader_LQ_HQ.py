import torch
import folders.folders_LQ_HQ as folders

class DataLoader(object):
    """Dataset class for IQA databases"""

    def __init__(self, dataset, path, img_indx, patch_size, patch_num, batch_size=1, istrain=True, self_patch_num=10):

        self.batch_size = batch_size
        self.istrain = istrain

        if dataset == 'live':
            self.data = folders.LIVEFolder(
                root=path, index=img_indx, patch_num=patch_num, patch_size = patch_size, self_patch_num=self_patch_num)
        elif dataset == 'csiq':
            self.data = folders.CSIQFolder(
                root=path, index=img_indx, patch_num=patch_num, patch_size = patch_size, self_patch_num=self_patch_num)
        elif dataset == 'kadid10k':
            self.data = folders.Kadid10kFolder(
                root=path, index=img_indx, patch_num=patch_num, patch_size = patch_size, self_patch_num=self_patch_num)
        elif dataset == 'tid2013':
            self.data = folders.TID2013Folder(
                root=path, index=img_indx, patch_num=patch_num, patch_size = patch_size, self_patch_num=self_patch_num)

    def get_dataloader(self):
        if self.istrain:
            dataloader = torch.utils.data.DataLoader(
                self.data, batch_size=self.batch_size, shuffle=True)
        else:
            dataloader = torch.utils.data.DataLoader(
                self.data, batch_size=1, shuffle=False)
        return dataloader
