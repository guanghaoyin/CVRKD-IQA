import torch
import os
from option_train_DistillationIQA import set_args, check_args
import numpy as np
from models.DistillationIQA import DistillationIQANet
from PIL import Image
import torchvision

img_num = {
        'kadid10k': list(range(0,10125)),
        'live': list(range(0, 29)),#ref HR image
        'csiq': list(range(0, 30)),#ref HR image
        'tid2013': list(range(0, 25)),
        'livec': list(range(0, 1162)),# no-ref image
        'koniq-10k': list(range(0, 10073)),# no-ref image
        'bid': list(range(0, 586)),# no-ref image
    }
folder_path = {
        'pipal':'./dataset/PIPAL',
        'live': './dataset/LIVE/',
        'csiq': './dataset/CSIQ/',
        'tid2013': './dataset/TID2013/',
        'livec': './dataset/LIVEC/',
        'koniq-10k': './dataset/koniq-10k/',
        'bid': './dataset/BID/',
        'kadid10k':'./dataset/kadid10k/'
    }


class DistillationIQASolver(object):
    def __init__(self, config, lq_path, ref_path):
        self.config = config
        self.config.teacherNet_model_path = './model_zoo/FR_teacher_cross_dataset.pth'
        self.config.studentNet_model_path = './model_zoo/NAR_student_cross_dataset.pth'

        self.device = torch.device('cuda' if config.gpu_ids is not None else 'cpu')
        self.txt_log_path = os.path.join(config.log_checkpoint_dir,'log.txt')
        with open(self.txt_log_path,"w+") as f:
            f.close()
        
        #model
        self.teacherNet = DistillationIQANet(self_patch_num=config.self_patch_num, distillation_layer=config.distillation_layer)
        if config.teacherNet_model_path:
            self.teacherNet._load_state_dict(torch.load(config.teacherNet_model_path))
        self.teacherNet = self.teacherNet.to(self.device)
        self.teacherNet.train(False)
        self.studentNet = DistillationIQANet(self_patch_num=config.self_patch_num, distillation_layer=config.distillation_layer)
        if config.studentNet_model_path:
            self.studentNet._load_state_dict(torch.load(config.studentNet_model_path))
        self.studentNet = self.studentNet.to(self.device)
        self.studentNet.train(True)

        self.transform = torchvision.transforms.Compose([
                    torchvision.transforms.RandomCrop(size=self.config.patch_size),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                     std=(0.229, 0.224, 0.225))
                ])
        #data
        self.LQ_patches = self.preprocess(lq_path)
        self.ref_patches = self.preprocess(ref_path)
    
    def preprocess(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            img= img.convert('RGB')
        patches = []
        for _ in range(self.config.self_patch_num):
            patch = self.transform(img)
            patches.append(patch.unsqueeze(0))
        patches = torch.cat(patches, 0)
        return patches.unsqueeze(0)

    def test(self):
        self.studentNet.train(False)
        LQ_patches, ref_patches = self.LQ_patches.to(self.device), self.ref_patches.to(self.device)
        with torch.no_grad():
            _, _, pred = self.studentNet(LQ_patches, ref_patches)
        return float(pred.item())

if __name__ == "__main__":
    config = set_args()
    config = check_args(config)

    lq_path = './dataset/koniq-10k/1024x768/28311109.jpg'
    ref_path = './dataset/DIV2K_ref/val_HR/0801.png'
    label = 1.15686274509804
    solver = DistillationIQASolver(config=config, lq_path=lq_path, ref_path=ref_path)
    scores = []
    for _ in range(10):
        scores.append(solver.test())
    print(np.mean(scores))
    # result 1.2577123641967773







    
