import torch
import torchvision
import torch.nn.functional as f

import math
from tqdm import tqdm

from utils import radam
import utils.viz_utils as viz_utils

from models.style_networks import SemSegE2VID
from models.deeplab import Net
from models.patchmlp import PatchSampleF
from models.patchnce import PatchSimLoss

import training.base_trainer
from evaluation.metrics import MetricsSemseg
from utils.loss_functions import TaskLoss
from utils.viz_utils import plot_confusion_matrix

from e2vid.utils.loading_utils import load_model
from e2vid.image_reconstructor import ImageReconstructor
from models.unet import unet
from models import resnet38d
import torch.nn.functional as F
import pdb
from utils.cls_opt import PolyOptimizer, cam_on_image
import numpy as np
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix
from e2vid.model import unet
from matplotlib import pyplot as plt
        
#python train.py --settings_file config/settings_DSEC_ours_weak.yaml

class OursSupervisedModel(training.base_trainer.BaseTrainer):
    def __init__(self, settings, train=True):
        self.is_training = train
        super(OursSupervisedModel, self).__init__(settings)
        self.do_val_training_epoch = False
        self.right_count = 0
        self.wrong_count = 0
        self.categories = ['BG', 'building','fence','person','pole','road',
                                           'sidewalk','vegetation','car','wall','traffic sign']
        
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        self.invTrans = transforms.Compose([transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                     std = [ 1., 1., 1. ]),
                               ])
        
        self.global_step = 0
        self.num_classes = self.settings.semseg_num_classes


    def init_fn(self):
        # self.buildModels()
        self.createOptimizerDict()

        
    def init_loss(self):
        # Decoder Loss
        self.cycle_content_loss = torch.nn.L1Loss()
        self.cycle_attribute_loss = torch.nn.L1Loss()

        # Task Loss
        self.task_loss = TaskLoss(losses=self.settings.task_loss, gamma=2.0, num_classes=self.settings.semseg_num_classes,
                                  ignore_index=self.settings.semseg_ignore_label, reduction='mean')
        self.train_statistics = {}

        self.metrics_semseg_b = MetricsSemseg(self.settings.semseg_num_classes, self.settings.semseg_ignore_label,
                                              self.settings.semseg_class_names)
        
    def buildModels(self):
        
        # Sim Loss
        self.criterionNCE = []
        for nce_layer in range(4):
            self.criterionNCE.append(PatchSimLoss().to(self.device))

        # Front End Sensor B


        self.input_height = math.ceil(self.settings.img_size_b[0] / 8.0) * 8
        self.input_width = math.ceil(self.settings.img_size_b[1] / 8.0) * 8
        
        self.unet1 = unet.UNetRecurrentEncDistillMulti(num_input_channels=self.settings.input_channels_b,
                         num_output_channels=1,
                         recurrent_block_type='convlstm',
                         skip_type='sum',
                         activation='sigmoid',
                         num_encoders=3,
                         base_num_channels=32,
                         num_residual_blocks=2,
                         norm='BN',
                         use_upsample_conv=False)
        
        self.unet2 = unet.UNetRecurrentEncDistillMulti(num_input_channels=self.settings.input_channels_b,
                         num_output_channels=1,
                         recurrent_block_type='convlstm',
                         skip_type='sum',
                         activation='sigmoid',
                         num_encoders=3,
                         base_num_channels=32,
                         num_residual_blocks=2,
                         norm='BN',
                         use_upsample_conv=False)
        
        self.models_dict = {"enc1": self.unet1, "enc2": self.unet2}
        self.task_backend1 = SemSegE2VID(input_c=256, output_c=self.settings.semseg_num_classes,
                                        skip_connect=self.settings.skip_connect_task,
                                        skip_type=self.settings.skip_connect_task_type)

        self.task_backend2 = SemSegE2VID(input_c=256, output_c=self.settings.semseg_num_classes,
                                        skip_connect=self.settings.skip_connect_task,
                                        skip_type=self.settings.skip_connect_task_type)
        
        self.models_dict["dec1"] = self.task_backend1
        self.models_dict["dec2"] = self.task_backend2
        
        
        self.num_classes = self.settings.semseg_num_classes
        
        self.proto_queue_A2 = [[] for i in range(self.num_classes)]
        self.proto_queue_B2 = [[] for i in range(self.num_classes)]
        self.proto_queue_A4 = [[] for i in range(self.num_classes)]
        self.proto_queue_B4 = [[] for i in range(self.num_classes)]
        self.proto_queue_A8 = [[] for i in range(self.num_classes)]
        self.proto_queue_B8 = [[] for i in range(self.num_classes)]

        self.global_step = 0
        self.step_count = 0
        self.infer_count = 0


    def createOptimizerDict(self):
        """Creates the dictionary containing the optimizer for the the specified subnetworks"""
        if not self.is_training:
            self.optimizers_dict = {}
            return

        
        optimizer_back = radam.RAdam([
                                        {'params': self.unet1.parameters()},
                                        {'params': self.task_backend1.parameters()},
                                        {'params': self.unet2.parameters()},
                                        {'params': self.task_backend2.parameters()},
                                    ],
                                     lr=self.settings.lr_back,
                                     weight_decay=0.,
                                     betas=(0., 0.999))
        
        
        self.optimizers_dict = {"optimizer_back": optimizer_back}

    def trainEpoch(self):
        self.pbar = tqdm(total=self.train_loader_sensor_b.__len__(), unit='Batch', unit_scale=True)
        for model in self.models_dict:
            self.models_dict[model].train()

        for i_batch, sample_batched in enumerate(self.train_loader_sensor_b):
            out = self.train_step(sample_batched)

            self.train_summaries(out[0])

            self.step_count += 1
            self.pbar.set_postfix(TrainLoss='{:.2f}'.format(out[-1].data.cpu().numpy()))
            self.pbar.update(1)
        self.pbar.close()

    def train_step(self, input_batch):
        # Task Step
        # optimizers_list = []
        # optimizers_list.append('optimizer_back')
 
        for key_word in self.optimizers_dict:
            optimizer_key_word = self.optimizers_dict[key_word]
            optimizer_key_word.zero_grad()

        d_final_loss, d_losses, d_outputs = self.task_train_step(input_batch)

        d_final_loss.backward()
        for key_word in self.optimizers_dict:
            optimizer_key_word = self.optimizers_dict[key_word]
            optimizer_key_word.step()

        return d_losses, d_outputs, d_final_loss


    def train_init_step(self, input_batch):
        for model in self.models_dict:
            self.models_dict[model].train()
        self.task_train_step(input_batch, init_func=True)

        return

    def task_train_step(self, batch, init_func=False):

        # data_b = batch[0].to(self.device)
        data_A = batch[0].cuda()
        # data_b_short = batch[3].to(self.device)
        data_B = batch[3].to(data_A.device)

        if self.settings.require_paired_data_train_b:
            labels_A = batch[2].to(data_A.device)
            dense_labels_A = batch[4].to(data_A.device)
            labels_B = labels_A
        else:
            labels_A = batch[1].to(data_A.device)
            dense_labels_A = batch[3].to(data_A.device)
            labels_B = labels_A
        

        # Set BatchNorm Statistics to Train
        for model in self.models_dict:
            self.models_dict[model].train()
            
        enc_net = self.models_dict["enc1"]
        task_backend = self.models_dict["dec1"]
        
        enc_net2 = self.models_dict["enc2"]
        task_backend2 = self.models_dict["dec2"]
        # img = batch[1].cuda()
        img = batch[1].to(data_A.device)
        B,_,H,W = img.size()        
   
        th = self.settings.dual_thres

        ####################################################################################################################
        ############################################### A ##################################################################
        ####################################################################################################################
        states_A = None
        for i in range(self.settings.nr_events_data_b):
            event_tensor = data_A[:, i * self.settings.input_channels_b:(i + 1) * self.settings.input_channels_b, :, :]
            states_A, latent_A = enc_net(event_tensor, states_A)
        pred_A = task_backend(latent_A)[1] # latent_real -> 1/8 scale (1/8 H x 1/8 W x), pred -> 1 scale (H x W x )
        _pred_A = F.softmax(F.relu(pred_A),dim=1)

        ## CLS 쳐내기
        num_cls = pred_A.shape[1]
        for i in range(labels_A.shape[0]):
            cls_temp_ = labels_A[i].unique()[:-1] # reject 255
            for c in range(num_cls):
                if c not in cls_temp_:
                    _pred_A[i,c] *= 0
        
        ignore_mask_A = torch.max(_pred_A,dim=1)[0]<th
        pred_A_lbl = torch.argmax(_pred_A,dim=1)
        pred_A_lbl[ignore_mask_A]=255
        
        
        # proto
        pt_th = 0.7

        ##########################
        proto_opts = [8]
        distill_opts = [2,4,8]

        if 2 in proto_opts:
            feat_A2 = F.interpolate(latent_A[2],size=labels_A.size()[1:],mode='bilinear',align_corners=False)
            _feat_A2 = F.normalize(feat_A2,dim=1)
            D2= feat_A2.size(1)
        
        if 4 in proto_opts:
            feat_A4 = F.interpolate(latent_A[4],size=labels_A.size()[1:],mode='bilinear',align_corners=False)
            _feat_A4 = F.normalize(feat_A4,dim=1)
            D4= feat_A4.size(1)
        
        if 8 in proto_opts:
            feat_A8 = F.interpolate(latent_A[8],size=labels_A.size()[1:],mode='bilinear',align_corners=False)
            _feat_A8 = F.normalize(feat_A8,dim=1)
            D8= feat_A8.size(1)
        

        gt_mask = torch.zeros((B,self.num_classes)).cuda()
        
        for i in range(labels_A.shape[0]):
            cls_temp_ = labels_A[i].unique()[:-1] # reject 255
            for c in cls_temp_:
               
                #Regional Prototype extraction
                conf_region = (_pred_A[i,c]*(_pred_A[i,c]>pt_th).float()).detach()

                if 2 in proto_opts:
                    proto2 = (_feat_A2[i].view(D2,-1).detach()@(conf_region).view(-1,1)).view(-1)/(conf_region.sum()+1e-5)  #D H W * HW 1 -> D
                    _proto2 = F.normalize(proto2,dim=-1)

                if 4 in proto_opts:
                    proto4 = (_feat_A4[i].view(D4,-1).detach()@(conf_region).view(-1,1)).view(-1)/(conf_region.sum()+1e-5)  #D H W * HW 1 -> D
                    _proto4 = F.normalize(proto4,dim=-1)

                if 8 in proto_opts:
                    proto8 = (_feat_A8[i].view(D8,-1).detach()@(conf_region).view(-1,1)).view(-1)/(conf_region.sum()+1e-5)  #D H W * HW 1 -> D
                    _proto8 = F.normalize(proto8,dim=-1)

                #Aggregate Prototype
                if conf_region.sum()>10:
                    if 2 in proto_opts:
                        self.proto_queue_A2[c].append(_proto2)
                    if 4 in proto_opts:
                        self.proto_queue_A4[c].append(_proto4)
                    if 8 in proto_opts:
                        self.proto_queue_A8[c].append(_proto8)
                    gt_mask[i,c]=1

        #Stack prototype in memory
        mem_len = 200
        for i in range(self.num_classes):
            
            if 2 in proto_opts and len(self.proto_queue_A2[i])>mem_len:
                self.proto_queue_A2[i] = self.proto_queue_A2[i][-mem_len:]
            if 4 in proto_opts and len(self.proto_queue_A4[i])>mem_len:
                self.proto_queue_A4[i] = self.proto_queue_A4[i][-mem_len:]
            if 8 in proto_opts and len(self.proto_queue_A8[i])>mem_len:
                self.proto_queue_A8[i] = self.proto_queue_A8[i][-mem_len:]
        
        if 2 in proto_opts:
            mean_protos_A2 = torch.zeros((self.num_classes,D2)).cuda()
        if 4 in proto_opts:
            mean_protos_A4 = torch.zeros((self.num_classes,D4)).cuda()
        if 8 in proto_opts:
            mean_protos_A8 = torch.zeros((self.num_classes,D8)).cuda()

        for i in range(self.num_classes):
            if  2 in proto_opts:
                if len(self.proto_queue_A2[i])>10:
                    mean_protos_A2[i] = torch.stack(self.proto_queue_A2[i]).mean(0) #mem_len C D -> mean -> normalize #C D -> B C D
                else:
                    valid_A = 1
            if  4 in proto_opts:
                if len(self.proto_queue_A4[i])>10:
                    mean_protos_A4[i] = torch.stack(self.proto_queue_A4[i]).mean(0) #mem_len C D -> mean -> normalize #C D -> B C D
                else:
                    valid_A = 1
            if  8 in proto_opts :
                if len(self.proto_queue_A8[i])>10:
                    mean_protos_A8[i] = torch.stack(self.proto_queue_A8[i]).mean(0) #mem_len C D -> mean -> normalize #C D -> B C D
                    valid_A = 1
                else: 
                    valid_A = 0

        if 2 in proto_opts:
            _mean_protos_A2 = F.normalize(mean_protos_A2,dim=-1,p=2).repeat(B,1,1) #C D -> B C D                
        if 4 in proto_opts:
            _mean_protos_A4 = F.normalize(mean_protos_A4,dim=-1,p=2).repeat(B,1,1) #C D -> B C D                
        if 8 in proto_opts:
            _mean_protos_A8 = F.normalize(mean_protos_A8,dim=-1,p=2).repeat(B,1,1) #C D -> B C D                

        temperature = 0.1

        ####################################################################################################################
        ############################################### B ##################################################################
        ####################################################################################################################
        states_B = None
        for i in range(self.settings.nr_events_data_b):
            event_tensor_short = data_B[:, i * self.settings.input_channels_b:(i + 1) * self.settings.input_channels_b, :, :]
            states_B, latent_B = enc_net2(event_tensor_short, states_B)
        pred_B = task_backend2(latent_B)[1]
        _pred_B = F.softmax(F.relu(pred_B),dim=1)

        ## CLS 쳐내기
        num_cls = pred_B.shape[1]
        for i in range(labels_B.shape[0]):
            cls_temp_ = labels_B[i].unique()[:-1] # reject 255
            for c in range(num_cls):
                if c not in cls_temp_:
                    _pred_B[i,c] *= 0

        ignore_mask_B = torch.max(_pred_B,dim=1)[0]<th
        pred_B_lbl = torch.argmax(_pred_B,dim=1)
        pred_B_lbl[ignore_mask_B]=255
        
        #############
        ####Proto####
        #############

        if 2 in proto_opts:
            feat_B2 = F.interpolate(latent_B[2],size=labels_B.size()[1:],mode='bilinear',align_corners=False)
            _feat_B2 = F.normalize(feat_B2,dim=1)
            D2= feat_B2.size(1)
        
        if 4 in proto_opts:
            feat_B4 = F.interpolate(latent_B[4],size=labels_B.size()[1:],mode='bilinear',align_corners=False)
            _feat_B4 = F.normalize(feat_B4,dim=1)
            D4= feat_B4.size(1)
        
        if 8 in proto_opts:
            feat_B8 = F.interpolate(latent_B[8],size=labels_B.size()[1:],mode='bilinear',align_corners=False)
            _feat_B8 = F.normalize(feat_B8,dim=1)
            D8= feat_B8.size(1)
        

        gt_mask = torch.zeros((B,self.num_classes)).cuda()
        
        for i in range(labels_B.shape[0]):
            cls_temp_ = labels_B[i].unique()[:-1] # reject 255
            for c in cls_temp_:
               
                #Regional Prototype extraction
                conf_region = (_pred_B[i,c]*(_pred_B[i,c]>pt_th).float()).detach()

                if 2 in proto_opts:
                    proto2 = (_feat_B2[i].view(D2,-1).detach()@(conf_region).view(-1,1)).view(-1)/(conf_region.sum()+1e-5)  #D H W * HW 1 -> D
                    _proto2 = F.normalize(proto2,dim=-1)

                if 4 in proto_opts:
                    proto4 = (_feat_B4[i].view(D4,-1).detach()@(conf_region).view(-1,1)).view(-1)/(conf_region.sum()+1e-5)  #D H W * HW 1 -> D
                    _proto4 = F.normalize(proto4,dim=-1)

                if 8 in proto_opts:
                    proto8 = (_feat_B8[i].view(D8,-1).detach()@(conf_region).view(-1,1)).view(-1)/(conf_region.sum()+1e-5)  #D H W * HW 1 -> D
                    _proto8 = F.normalize(proto8,dim=-1)

                #Aggregate Prototype
                if conf_region.sum()>10:
                    if 2 in proto_opts:
                        self.proto_queue_B2[c].append(_proto2)
                    if 4 in proto_opts:
                        self.proto_queue_B4[c].append(_proto4)
                    if 8 in proto_opts:
                        self.proto_queue_B8[c].append(_proto8)
                    gt_mask[i,c]=1

        #Stack prototype in memory
        for i in range(self.num_classes):
            
            if 2 in proto_opts and len(self.proto_queue_B2[i])>mem_len:
                self.proto_queue_B2[i] = self.proto_queue_B2[i][-mem_len:]
            if 4 in proto_opts and len(self.proto_queue_B4[i])>mem_len:
                self.proto_queue_B4[i] = self.proto_queue_B4[i][-mem_len:]
            if 8 in proto_opts and len(self.proto_queue_B8[i])>mem_len:
                self.proto_queue_B8[i] = self.proto_queue_B8[i][-mem_len:]
        
        if 2 in proto_opts:
            mean_protos_B2 = torch.zeros((self.num_classes,D2)).cuda()
        if 4 in proto_opts:
            mean_protos_B4 = torch.zeros((self.num_classes,D4)).cuda()
        if 8 in proto_opts:
            mean_protos_B8 = torch.zeros((self.num_classes,D8)).cuda()

    
        for i in range(self.num_classes):
            if  2 in proto_opts:
                if len(self.proto_queue_B2[i])>10:
                    mean_protos_B2[i] = torch.stack(self.proto_queue_B2[i]).mean(0) #mem_len C D -> mean -> normalize #C D -> B C D
                else:
                    valid_B = 1
            if  4 in proto_opts:
                if len(self.proto_queue_B4[i])>10:
                    mean_protos_B4[i] = torch.stack(self.proto_queue_B4[i]).mean(0) #mem_len C D -> mean -> normalize #C D -> B C D #############3
                else:
                    valid_B = 1
            if  8 in proto_opts :
                if len(self.proto_queue_B8[i])>10:
                    mean_protos_B8[i] = torch.stack(self.proto_queue_B8[i]).mean(0) #mem_len C D -> mean -> normalize #C D -> B C D
                    valid_B = 1
                else: 
                    valid_B = 0

        if 2 in proto_opts:
            _mean_protos_B2 = F.normalize(mean_protos_B2,dim=-1,p=2).repeat(B,1,1) #C D -> B C D                
        if 4 in proto_opts:
            _mean_protos_B4 = F.normalize(mean_protos_B4,dim=-1,p=2).repeat(B,1,1) #C D -> B C D                
        if 8 in proto_opts:
            _mean_protos_B8 = F.normalize(mean_protos_B8,dim=-1,p=2).repeat(B,1,1) #C D -> B C D   
        
        ###############################################
        #Proto_proj
        # protoproj_opts = [2,4,8] 
        ###############################################

        if 2 in proto_opts:
            _mean_protos_A2toB2 = self.unet1.proj_head_for_distill_2(_mean_protos_A2.permute(0,2,1).unsqueeze(-1)).permute(0,2,1,3).squeeze(-1) 
            _mean_protos_A2toB2 = F.normalize(_mean_protos_A2toB2,dim=-1)
            _mean_protos_B2toA2 = self.unet2.proj_head_for_distill_2(_mean_protos_B2.permute(0,2,1).unsqueeze(-1)).permute(0,2,1,3).squeeze(-1) 
            _mean_protos_B2toA2 = F.normalize(_mean_protos_B2toA2,dim=-1)
            
            # _mean_protos_A2_final = F.normalize((_mean_protos_A2 + _mean_protos_B2toA2)/2,dim=1,p=2)
            # _mean_protos_B2_final = F.normalize((_mean_protos_B2 + _mean_protos_A2toB2)/2,dim=1,p=2)

            sim_A2 = torch.bmm(_mean_protos_A2,_feat_A2.view(B,D2,-1)).view(B,self.num_classes,H,W) # B C D * B D HW
            nce_loss_A2 = F.cross_entropy(sim_A2/temperature,pred_B_lbl.detach(),ignore_index=255)
            sim_B2toA2 = torch.bmm(_mean_protos_B2toA2,_feat_A2.view(B,D2,-1)).view(B,self.num_classes,H,W) # B C D * B D HW
            nce_loss_B2toA2 = F.cross_entropy(sim_B2toA2/temperature,pred_B_lbl.detach(),ignore_index=255)
            
            sim_B2 = torch.bmm(_mean_protos_B2,_feat_B2.view(B,D2,-1)).view(B,self.num_classes,H,W) # B C D * B D HW
            nce_loss_B2 = F.cross_entropy(sim_B2/temperature,pred_A_lbl.detach(),ignore_index=255)
            sim_A2toB2 = torch.bmm(_mean_protos_A2toB2,_feat_B2.view(B,D2,-1)).view(B,self.num_classes,H,W) # B C D * B D HW
            nce_loss_A2toB2 = F.cross_entropy(sim_A2toB2/temperature,pred_A_lbl.detach(),ignore_index=255)

        if 4 in proto_opts:
            _mean_protos_A4toB4 = self.unet1.proj_head_for_distill_4(_mean_protos_A4.permute(0,2,1).unsqueeze(-1)).permute(0,2,1,3).squeeze(-1) 
            _mean_protos_A4toB4 = F.normalize(_mean_protos_A4toB4,dim=-1)
            _mean_protos_B4toA4 = self.unet2.proj_head_for_distill_4(_mean_protos_B4.permute(0,2,1).unsqueeze(-1)).permute(0,2,1,3).squeeze(-1) 
            _mean_protos_B4toA4 = F.normalize(_mean_protos_B4toA4,dim=-1)

            # _mean_protos_A4_final = F.normalize((_mean_protos_A4 + _mean_protos_B4toA4)/2,dim=1,p=2)
            # _mean_protos_B4_final = F.normalize((_mean_protos_B4 + _mean_protos_A4toB4)/2,dim=1,p=2)
        
            sim_A4 = torch.bmm(_mean_protos_A4,_feat_A4.view(B,D4,-1)).view(B,self.num_classes,H,W) # B C D * B D HW
            nce_loss_A4 = F.cross_entropy(sim_A4/temperature,pred_B_lbl.detach(),ignore_index=255)
            sim_B4toA4 = torch.bmm(_mean_protos_B4toA4,_feat_A4.view(B,D4,-1)).view(B,self.num_classes,H,W) # B C D * B D HW
            nce_loss_B4toA4 = F.cross_entropy(sim_B4toA4/temperature,pred_B_lbl.detach(),ignore_index=255)
            
            sim_B4 = torch.bmm(_mean_protos_B4,_feat_B4.view(B,D4,-1)).view(B,self.num_classes,H,W) # B C D * B D HW
            nce_loss_B4 = F.cross_entropy(sim_B4/temperature,pred_A_lbl.detach(),ignore_index=255)
            sim_A4toB4 = torch.bmm(_mean_protos_A4toB4,_feat_B4.view(B,D4,-1)).view(B,self.num_classes,H,W) # B C D * B D HW
            nce_loss_A4toB4 = F.cross_entropy(sim_A4toB4/temperature,pred_A_lbl.detach(),ignore_index=255)
            
        if 8 in proto_opts:
            _mean_protos_A8toB8 = self.unet1.proj_head_for_distill_8(_mean_protos_A8.permute(0,2,1).unsqueeze(-1)).permute(0,2,1,3).squeeze(-1) 
            _mean_protos_A8toB8 = F.normalize(_mean_protos_A8toB8,dim=-1)
            _mean_protos_B8toA8 = self.unet2.proj_head_for_distill_8(_mean_protos_B8.permute(0,2,1).unsqueeze(-1)).permute(0,2,1,3).squeeze(-1) 
            _mean_protos_B8toA8 = F.normalize(_mean_protos_B8toA8,dim=-1)
            
            # _mean_protos_A8_final = F.normalize((_mean_protos_A8 + _mean_protos_B8toA8)/2,dim=1,p=2)
            # _mean_protos_B8_final = F.normalize((_mean_protos_B8 + _mean_protos_A8toB8)/2,dim=1,p=2)

            sim_A8 = torch.bmm(_mean_protos_A8,_feat_A8.view(B,D8,-1)).view(B,self.num_classes,H,W) # B C D * B D HW
            nce_loss_A8 = F.cross_entropy(sim_A8/temperature,pred_B_lbl.detach(),ignore_index=255)
            sim_B8toA8 = torch.bmm(_mean_protos_B8toA8,_feat_A8.view(B,D8,-1)).view(B,self.num_classes,H,W) # B C D * B D HW
            nce_loss_B8toA8 = F.cross_entropy(sim_B8toA8/temperature,pred_B_lbl.detach(),ignore_index=255)
            
            sim_B8 = torch.bmm(_mean_protos_B8,_feat_B8.view(B,D8,-1)).view(B,self.num_classes,H,W) # B C D * B D HW
            nce_loss_B8 = F.cross_entropy(sim_B8/temperature,pred_A_lbl.detach(),ignore_index=255)
            sim_A8toB8 = torch.bmm(_mean_protos_A8toB8,_feat_B8.view(B,D8,-1)).view(B,self.num_classes,H,W) # B C D * B D HW
            nce_loss_A8toB8 = F.cross_entropy(sim_A8toB8/temperature,pred_A_lbl.detach(),ignore_index=255)
          
        
        ####################################################################################################################
        ############################################### losses #############################################################
        ####################################################################################################################       
        losses = {}
        outputs = {}
        loss_all = 0.
        loss_proto_A = 0.
        loss_proto_B = 0.

        loss_point_A = F.cross_entropy(pred_A, labels_A,ignore_index=255)
        loss_point_B = F.cross_entropy(pred_B,labels_B,ignore_index=255)
        loss_all += loss_point_A 
        loss_all += loss_point_B

        # To avoid NaN
        nan_flag = False
        for bidx in range(pred_A_lbl.shape[0]):
            if pred_A_lbl[bidx].min()==255 or pred_B_lbl[bidx].min()==255:
                nan_flag = True

        if nan_flag:
            loss_dual = torch.zeros_like(loss_point_A)
        else:
            loss_dual = F.cross_entropy(pred_A,pred_B_lbl.detach(),ignore_index=255) + F.cross_entropy(pred_B,pred_A_lbl.detach(),ignore_index=255)   
            loss_all += loss_dual * 0.5
       
        
        if self.epoch_count >= 2:
            if valid_A:
                if 2 in proto_opts:
                    loss_proto_A += 0.5*(nce_loss_A2/2 + nce_loss_B2toA2/2) 
                if 4 in proto_opts:
                    loss_proto_A += 0.5*(nce_loss_A4/2 + nce_loss_B4toA4/2) 
                if 8 in proto_opts:
                    loss_proto_A += 0.5*(nce_loss_A8/2 + nce_loss_B8toA8/2) 
                
                loss_all += loss_proto_A/len(proto_opts)

            if valid_B:
                if 2 in proto_opts:
                    loss_proto_B += 0.5*(nce_loss_B2/2 + nce_loss_A2toB2/2) 
                if 4 in proto_opts:
                    loss_proto_B += 0.5*(nce_loss_B4/2 + nce_loss_A4toB4/2) 
                if 8 in proto_opts:
                    loss_proto_B += 0.5*(nce_loss_B8/2 + nce_loss_A8toB8/2) 
                
                loss_all += loss_proto_B/len(proto_opts)

        # print('\n')
        # print(valid_A,valid_B)
        # print(loss_proto_A)
        # print(loss_proto_B)

        ################################################################        
        ## Distillation
        ################################################################
                        
       

        loss_distill = 0.

        if 2 in distill_opts:
            feat_A2 = latent_A[2]
            feat_B2 = latent_B[2]
            _feat_A2 = F.normalize(feat_A2,dim=1)
            _feat_B2 = F.normalize(feat_B2,dim=1)
            _feat_AB2 = F.normalize(self.unet1.proj_head_for_distill_2(_feat_A2),dim=1)
            _feat_BA2 = F.normalize(self.unet2.proj_head_for_distill_2(_feat_B2),dim=1)
            loss_distill += F.l1_loss(_feat_AB2, _feat_B2.detach()) + F.l1_loss(_feat_BA2, _feat_A2.detach())

        if 4 in distill_opts: 
            feat_A4 = latent_A[4]
            feat_B4 = latent_B[4]
            _feat_A4 = F.normalize(feat_A4,dim=1)
            _feat_B4 = F.normalize(feat_B4,dim=1)
            _feat_AB4 = F.normalize(self.unet1.proj_head_for_distill_4(_feat_A4),dim=1)
            _feat_BA4 = F.normalize(self.unet2.proj_head_for_distill_4(_feat_B4),dim=1)
            loss_distill += F.l1_loss(_feat_AB4, _feat_B4.detach()) + F.l1_loss(_feat_BA4, _feat_A4.detach())

        if 8 in distill_opts:
            feat_A8 = latent_A[8]
            feat_B8 = latent_B[8]
            _feat_A8 = F.normalize(feat_A8,dim=1)
            _feat_B8 = F.normalize(feat_B8,dim=1)
            _feat_AB8 = F.normalize(self.unet1.proj_head_for_distill_8(_feat_A8),dim=1)
            _feat_BA8 = F.normalize(self.unet2.proj_head_for_distill_8(_feat_B8),dim=1)
            loss_distill += F.l1_loss(_feat_AB8, _feat_B8.detach()) + F.l1_loss(_feat_BA8, _feat_A8.detach())
        
        if self.epoch_count >= 2:
            loss_all += loss_distill/(len(distill_opts)*2)

        # if self.visualize_epoch():
        if self.global_step %50==0:
        # if True:
            self.visTaskStep(data_A, data_B, pred_A, pred_B, dense_labels_A, batch[1].to(self.device), img, _pred_A)

            
        losses['loss_seg_A'] = loss_point_A.detach()
        losses['loss_seg_B'] = loss_point_B.detach()
        if valid_A and self.epoch_count>=2:
            losses['loss_proto_A'] = loss_proto_A.detach()
        else:
            losses['loss_proto_A'] = loss_proto_A

        if valid_B and self.epoch_count>=2:
            losses['loss_proto_B'] = loss_proto_B.detach()
        else:
            losses['loss_proto_B'] = loss_proto_B
    
        losses['loss_dual'] = loss_dual.detach()
        losses['loss_distill'] = loss_distill.detach()
        
        self.global_step += 1

        return loss_all, losses, outputs


    def visTaskStep(self, data, data_short, pred, pred_short, labels, img_fake, img, cam):
        # pred = pred[1]
        if self.settings.random_crop:
            pred = F.interpolate(
                pred, size=([self.settings.crop_size[0], self.settings.crop_size[1]]), mode="bilinear", align_corners=True
            )
        else:
            pred = F.interpolate(
                pred, size=(img.size()[2:]), mode="bilinear", align_corners=True
            )
        pred_lbl = pred.argmax(dim=1)

        semseg = viz_utils.prepare_semseg(pred_lbl, self.settings.semseg_color_map,
                                              self.settings.semseg_ignore_label)
        if self.settings.random_crop:
            pred_short = F.interpolate(
                pred_short, size=([self.settings.crop_size[0], self.settings.crop_size[1]]), mode="bilinear", align_corners=True
            )
        else:
            pred_short = F.interpolate(
                pred_short, size=(img.size()[2:]), mode="bilinear", align_corners=True
            )
            
        pred_lbl_short = pred_short.argmax(dim=1)

        semseg_short = viz_utils.prepare_semseg(pred_lbl_short, self.settings.semseg_color_map,
                                              self.settings.semseg_ignore_label)
        
        
        
        semseg_gt = viz_utils.prepare_semseg(labels, self.settings.semseg_color_map, self.settings.semseg_ignore_label)

        nrow = 1
        viz_tensors = torch.cat(
            (viz_utils.createRGBImage(data[:nrow], separate_pol=self.settings.separate_pol_b).clamp(0, 1).to(
                self.device),
             viz_utils.createRGBImage(semseg[:nrow].to(self.device)),
             viz_utils.createRGBImage(data_short[:nrow], separate_pol=self.settings.separate_pol_b).clamp(0, 1).to(
                self.device),
             viz_utils.createRGBImage(semseg_short[:nrow].to(self.device)),
             viz_utils.createRGBImage(semseg_gt[:nrow].to(self.device)),
            #  viz_utils.createRGBImage(img_fake[:nrow]),
            #  viz_utils.createRGBImage(torch.from_numpy(cam).permute(2,0,1)[::1,:,:].unsqueeze(0).cuda()),
             ), dim=0)
        rgb_grid = torchvision.utils.make_grid(viz_tensors, nrow=nrow)
        self.img_summaries('train/semseg_event', rgb_grid, self.step_count)
        
        idx= 0
        
        for i in torch.unique(pred_lbl[idx]):
            temp = cam_on_image(self.invTrans(img[idx]).detach().cpu().numpy(),cam[idx,i].detach().cpu().numpy())
            self.cam_summaries('train/cam_'+self.categories[i],temp, self.step_count)

    def validationEpochs(self):
        self.resetValidationStatistics()

        with torch.no_grad():
            for model in self.models_dict:
                self.models_dict[model].eval()

            self.validationEpoch(self.val_loader_sensor_b, 'sensor_b')

            if len(self.validation_embeddings) != 0:
                self.saveEmbeddingSpace()

            if self.do_val_training_epoch:
                self.trainDatasetStatisticsEpoch('sensor_b', self.train_loader_sensor_b)

            self.resetValidationStatistics()

        self.pbar.close()

    def validationEpoch(self, data_loader, sensor_name):
        val_dataset_length = data_loader.__len__()
        self.pbar = tqdm(total=val_dataset_length, unit='Batch', unit_scale=True)
        tqdm.write("Validation on " + sensor_name)
        cumulative_losses = {}
        total_nr_steps = None

        for i_batch, sample_batched in enumerate(data_loader):
            self.validationBatchStep(sample_batched, sensor_name, i_batch, cumulative_losses, val_dataset_length)
            self.pbar.update(1)
            total_nr_steps = i_batch

        if sensor_name == 'sensor_a':
            metrics_semseg_a = self.metrics_semseg_a.get_metrics_summary()
            metric_semseg_a_mean_iou = metrics_semseg_a['mean_iou']
            cumulative_losses['semseg_sensor_a_mean_iou'] = metric_semseg_a_mean_iou
            metric_semseg_a_acc = metrics_semseg_a['acc']
            cumulative_losses['semseg_sensor_a_acc'] = metric_semseg_a_acc
            metrics_semseg_a_cm = metrics_semseg_a['cm']
            figure_semseg_a_cm = plot_confusion_matrix(metrics_semseg_a_cm, classes=self.settings.semseg_class_names,
                                                       normalize=True,
                                                       title='Normalized confusion matrix')
            self.summary_writer.add_figure('val_gray/semseg_cm',
                                           figure_semseg_a_cm, self.epoch_count)
            
        else:
            metrics_semseg_b = self.metrics_semseg_b.get_metrics_summary()
            metric_semseg_b_mean_iou = metrics_semseg_b['mean_iou']
            cumulative_losses['semseg_sensor_b_mean_iou'] = metric_semseg_b_mean_iou
            metric_semseg_b_acc = metrics_semseg_b['acc']
            cumulative_losses['semseg_sensor_b_acc'] = metric_semseg_b_acc
            metrics_semseg_b_cm = metrics_semseg_b['cm']
            figure_semseg_b_cm = plot_confusion_matrix(metrics_semseg_b_cm, classes=self.settings.semseg_class_names,
                                                       normalize=True,
                                                       title='Normalized confusion matrix')
            print("mIoU:",metric_semseg_b_mean_iou)
            self.summary_writer.add_figure('val_events/semseg_cm',
                                           figure_semseg_b_cm, self.epoch_count)

        self.val_summaries(cumulative_losses, total_nr_steps + 1)
        self.pbar.close()
        if self.val_confusion_matrix.sum() != 0:
            self.addValidationMatrix(sensor_name)

        self.saveValStatistics('val', sensor_name)

    def val_step(self, input_batch, sensor, i_batch, vis_reconstr_idx):
        """Calculates the performance measurements based on the input"""
        data = input_batch[0]
        paired_data = None
        if sensor == 'sensor_a':
            if self.settings.require_paired_data_val_a:
                paired_data = input_batch[1]
                labels = input_batch[2]
            else:
                labels = input_batch[1]
        else:
            if self.settings.require_paired_data_val_b:
                paired_data = input_batch[1]
                if self.settings.dataset_name_b == 'DDD17_events':
                    labels = input_batch[3]
                else:
                    labels = input_batch[2]
            else:
                labels = input_batch[1]


        losses = {}
        
     
        pred_lbl = self.valTaskStep(data, labels, losses, sensor)

        # self.saveSensorB(data, pred_lbl, labels, paired_data, self.infer_count, sensor)
        # self.infer_count += 1

        if vis_reconstr_idx != -1:
        # if True:
            self.visualizeSensorB(data, pred_lbl,
                                              labels, paired_data, vis_reconstr_idx, sensor)
        return losses, None

    def valTaskStep(self, data, labels, losses, sensor):
        """Computes the task loss and updates metrics"""
        # task_backend = self.models_dict["back_end"]
        enc_net = self.models_dict["enc1"]
        task_backend = self.models_dict["dec1"]
        
        states_real = None
        for i in range(self.settings.nr_events_data_b):
            event_tensor = data[:, i * self.settings.input_channels_b:(i + 1) * self.settings.input_channels_b, :, :]
            states_real, latent_real = enc_net(event_tensor, states_real)
        pred = task_backend(latent_real)[1]
        
        pred = F.interpolate(
            pred, size=(self.settings.img_size_b), mode="bilinear", align_corners=True
        )
        pred_lbl = pred.argmax(dim=1)

        loss_pred = self.task_loss(pred, target=labels)
        losses['semseg_' + sensor + '_loss'] = loss_pred.detach()
        
        if sensor == 'sensor_a':
            self.metrics_semseg_a.update_batch(pred_lbl, labels)
        else:
            self.metrics_semseg_b.update_batch(pred_lbl, labels)

        return pred_lbl

    def visualizeSensorB(self, data, pred_lbl, labels, paired_data,
                                 vis_reconstr_idx, sensor):
        nrow = 4
        vis_tensors = [viz_utils.createRGBImage(data[:nrow, -self.settings.input_channels_b:], separate_pol=self.settings.separate_pol_b).clamp(0, 1).to(
            self.device)]

        

        labels = f.interpolate(labels.float().unsqueeze(1), size=(self.input_height, self.input_width), mode='nearest').squeeze(1).long()

        semseg = viz_utils.prepare_semseg(pred_lbl, self.settings.semseg_color_map, self.settings.semseg_ignore_label)
        vis_tensors.append(viz_utils.createRGBImage(semseg[:nrow].to(self.device)))
        semseg_gt = viz_utils.prepare_semseg(labels, self.settings.semseg_color_map, self.settings.semseg_ignore_label)
        vis_tensors.append(viz_utils.createRGBImage(semseg_gt[:nrow]).to(self.device))
        vis_tensors.append(viz_utils.createRGBImage(self.invTrans(paired_data[:nrow])).to(self.device))
        rgb_grid = torchvision.utils.make_grid(torch.cat(vis_tensors, dim=0), nrow=nrow)
        self.img_summaries('val_' + sensor + '/reconst_input_' + sensor + '_' + str(vis_reconstr_idx),
                           rgb_grid, self.epoch_count)

    def saveSensorB(self, data, pred_lbl, labels, paired_data,
                                 vis_reconstr_idx, sensor):
        nrow = 4
        vis_tensors = [viz_utils.visualizeVoxelGrid(data[:nrow, -3:], separate_pol=self.settings.separate_pol_b).clamp(0, 1).to(
            self.device)]

        

        labels = f.interpolate(labels.float().unsqueeze(1), size=(self.input_height, self.input_width), mode='nearest').squeeze(1).long()

        semseg = viz_utils.prepare_semseg(pred_lbl, self.settings.semseg_color_map, self.settings.semseg_ignore_label)
        vis_tensors.append(viz_utils.createRGBImage(semseg[:nrow].to(self.device)))
        semseg_gt = viz_utils.prepare_semseg(labels, self.settings.semseg_color_map, self.settings.semseg_ignore_label)
        vis_tensors.append(viz_utils.createRGBImage(semseg_gt[:nrow]).to(self.device))
        vis_tensors.append(viz_utils.createRGBImage(self.invTrans(paired_data[:nrow])).to(self.device))
        rgb_grid = torchvision.utils.make_grid(torch.cat(vis_tensors, dim=0), nrow=nrow)

        plt.imsave('temp.png', rgb_grid.permute(1,2,0).cpu().detach().numpy())
        # plt.imsave('/ssd5/scripts_new/240307_dsec_night_infer_ours_final/'+str(vis_reconstr_idx).zfill(4)+'.png', rgb_grid.permute(1,2,0).cpu().detach().numpy())

    def resetValidationStatistics(self):
        self.metrics_semseg_b.reset()


                
    def count_rw(self, label, out):
       
    
        for b in range(label.size(0)):  
            gt = label[b].cpu().detach().numpy()
            gt_cls = np.nonzero(gt)[0]
            num = len(np.nonzero(gt)[0])
            pred = out[b].cpu().detach().numpy()
            pred_cls = pred.argsort()[-num:][::-1]

            for c in gt_cls:
                if c in pred_cls:
                    self.right_count += 1
                else:
                    self.wrong_count += 1
                    
        acc = (self.right_count*100)/(self.right_count+self.wrong_count+1e-5)
        return acc
    
    def max_norm(self, cam_cp):
        N, C, H, W = cam_cp.size()
        cam_cp = F.relu(cam_cp)
        max_v = torch.max(cam_cp.view(N, C, -1), dim=-1)[0].view(N, C, 1, 1)
        min_v = torch.min(cam_cp.view(N, C, -1), dim=-1)[0].view(N, C, 1, 1)
        cam_cp = F.relu(cam_cp - min_v - 1e-5) / (max_v - min_v + 1e-5)
        return cam_cp