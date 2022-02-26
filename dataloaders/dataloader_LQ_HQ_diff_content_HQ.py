import torch
import torchvision
import folders.folders_LQ_HQ_diff_content_HQ as folders

class DataLoader(object):
    """Dataset class for IQA databases"""

    def __init__(self, dataset, path, ref_path, img_indx, patch_size, patch_num, batch_size=1, istrain=True, self_patch_num=10, use_HQref = True):

        self.batch_size = batch_size
        self.istrain = istrain

        if (dataset == 'live') | (dataset == 'csiq') | (dataset == 'tid2013') | (dataset == 'livec') | (dataset == 'kadid10k'):
            # Train transforms
            if istrain:
                HQ_diff_content_transform = torchvision.transforms.Compose([
                    torchvision.transforms.RandomCrop(size=patch_size),
                    torchvision.transforms.RandomHorizontalFlip(),
                    torchvision.transforms.RandomVerticalFlip(),
                    torchvision.transforms.RandomRotation(degrees=180),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                     std=(0.229, 0.224, 0.225))
                ])
            # Test transforms
            else:
                HQ_diff_content_transform = torchvision.transforms.Compose([
                    torchvision.transforms.RandomCrop(size=patch_size),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                     std=(0.229, 0.224, 0.225))
                ])
        elif dataset == 'koniq-10k':
            if istrain:
                HQ_diff_content_transform = torchvision.transforms.Compose([
                    torchvision.transforms.RandomCrop(size=patch_size),
                    torchvision.transforms.RandomHorizontalFlip(),
                    torchvision.transforms.RandomVerticalFlip(),
                    torchvision.transforms.RandomRotation(degrees=180),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                     std=(0.229, 0.224, 0.225))])
            else:
                HQ_diff_content_transform = torchvision.transforms.Compose([
                    torchvision.transforms.RandomCrop(size=patch_size),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                     std=(0.229, 0.224, 0.225))])
        elif dataset == 'bid':
            if istrain:
                HQ_diff_content_transform = torchvision.transforms.Compose([
                    torchvision.transforms.Resize((512, 512)),
                    torchvision.transforms.RandomCrop(size=patch_size),
                    torchvision.transforms.RandomHorizontalFlip(),
                    torchvision.transforms.RandomVerticalFlip(),
                    torchvision.transforms.RandomRotation(degrees=180),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                     std=(0.229, 0.224, 0.225))])
            else:
                HQ_diff_content_transform = torchvision.transforms.Compose([
                    torchvision.transforms.Resize((512, 512)),
                    torchvision.transforms.RandomCrop(size=patch_size),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                     std=(0.229, 0.224, 0.225))])
        else:
             HQ_diff_content_transform = torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                     std=(0.229, 0.224, 0.225))])
        transforms = torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                     std=(0.229, 0.224, 0.225))])

        if dataset == 'live':
            self.data = folders.LIVEFolder(
                root=path, HQ_diff_content_root=ref_path, index=img_indx, transform=transforms, HQ_diff_content_transform=HQ_diff_content_transform, patch_num=patch_num, patch_size = patch_size, self_patch_num=self_patch_num)
        elif dataset == 'csiq':
            self.data = folders.CSIQFolder(
                root=path, HQ_diff_content_root=ref_path, index=img_indx, transform=transforms, HQ_diff_content_transform=HQ_diff_content_transform, patch_num=patch_num, patch_size = patch_size, self_patch_num=self_patch_num)
        elif dataset == 'kadid10k':
            self.data = folders.Kadid10kFolder(
                root=path, HQ_diff_content_root=ref_path, index=img_indx, transform=transforms, HQ_diff_content_transform=HQ_diff_content_transform, patch_num=patch_num, patch_size = patch_size, self_patch_num=self_patch_num)
        elif dataset == 'tid2013':
            self.data = folders.TID2013Folder(
                root=path, HQ_diff_content_root=ref_path, index=img_indx, transform=transforms, HQ_diff_content_transform=HQ_diff_content_transform, patch_num=patch_num, patch_size = patch_size, self_patch_num=self_patch_num)
        elif dataset == 'koniq-10k':
            self.data = folders.Koniq_10kFolder(
                root=path, HQ_diff_content_root=ref_path, index=img_indx, transform=transforms, HQ_diff_content_transform=HQ_diff_content_transform, patch_num=patch_num, patch_size = patch_size, self_patch_num=self_patch_num)
        elif dataset == 'livec':
            self.data = folders.LIVEChallengeFolder(
                root=path, HQ_diff_content_root=ref_path, index=img_indx, transform=transforms, HQ_diff_content_transform=HQ_diff_content_transform, patch_num=patch_num, patch_size = patch_size, self_patch_num=self_patch_num)

    def get_dataloader(self):
        if self.istrain:
            dataloader = torch.utils.data.DataLoader(
                self.data, batch_size=self.batch_size, shuffle=True)
        else:
            dataloader = torch.utils.data.DataLoader(
                self.data, batch_size=self.batch_size, shuffle=False)
        return dataloader
