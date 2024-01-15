import numpy
import torch.distributed as dist
import torch
import clip
import os

def reduce_tensor(tensor, n=None):
    if n is None:
        n = dist.get_world_size()
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt = rt / n
    
    return rt

def calc_mca(conf_matrix):
    per_class_num_correct = conf_matrix.diag()
    per_class_total_counts = conf_matrix.sum(dim=1) # compress along the columns (sum each row) to get total counts per class

    num_classes = per_class_num_correct.size(0)
    
    per_class_accuracies = torch.zeros(num_classes).cuda()
    non_zero_mask = per_class_total_counts > 0

    if torch.any(non_zero_mask):
        per_class_accuracies[non_zero_mask] = per_class_num_correct[non_zero_mask] / per_class_total_counts[non_zero_mask]    

    mca = per_class_accuracies.mean()

    return mca * 100

class MetricMeterMCA:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.confusion_matrix = torch.zeros((num_classes,num_classes)).cuda()
        self.mca = 0

    def update(self, conf_matrix):
        self.confusion_matrix += conf_matrix
        self.mca = calc_mca(self.confusion_matrix)

    def sync(self):
        self.confusion_matrix = reduce_tensor(self.confusion_matrix, 1)
        self.mca = calc_mca(self.confusion_matrix)
        
def reduce_tensor(tensor, n=None):
    if n is None:
        n = dist.get_world_size()
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt = rt / n
    return rt
   

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def sync(self):
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        val = torch.tensor(self.val).cuda()
        sum_v = torch.tensor(self.sum).cuda()
        count = torch.tensor(self.count).cuda()
        self.val = reduce_tensor(val, world_size).item()
        self.sum = reduce_tensor(sum_v, 1).item()
        self.count = reduce_tensor(count, 1).item()
        self.avg = self.sum / self.count


def epoch_saving(config, epoch, model,  max_accuracy, optimizer, lr_scheduler, logger, working_dir, is_best):
    save_state = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'lr_scheduler': lr_scheduler.state_dict(),
                  'max_accuracy': max_accuracy,
                  'epoch': epoch,
                  'config': config}
    
    save_path = os.path.join(working_dir, f'ckpt_epoch_{epoch}.pth')
    logger.info(f"{save_path} saving......")
    torch.save(save_state, save_path)
    logger.info(f"{save_path} saved !!!")
    if is_best:
        best_path = os.path.join(working_dir, f'best.pth')
        torch.save(save_state, best_path)
        logger.info(f"{best_path} saved !!!")


# def load_checkpoint(config, model, optimizer, lr_scheduler, logger):
#     if os.path.isfile(config.MODEL.RESUME): 
#         logger.info(f"==============> Resuming form {config.MODEL.RESUME}....................")
#         checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')
#         load_state_dict = checkpoint['model']

#         # now remove the unwanted keys:
#         if "module.prompt_learner.token_prefix" in load_state_dict:
#             del load_state_dict["module.prompt_learner.token_prefix"]

#         if "module.prompt_learner.token_suffix" in load_state_dict:
#             del load_state_dict["module.prompt_learner.token_suffix"]

#         if "module.prompt_learner.complete_text_embeddings" in load_state_dict:
#             del load_state_dict["module.prompt_learner.complete_text_embeddings"]

#         msg = model.load_state_dict(load_state_dict, strict=False)
#         logger.info(f"resume model: {msg}")

#         try:
#             optimizer.load_state_dict(checkpoint['optimizer'])
#             lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

#             start_epoch = checkpoint['epoch'] + 1
#             max_accuracy = checkpoint['max_accuracy']

#             logger.info(f"=> loaded successfully '{config.MODEL.RESUME}' (epoch {checkpoint['epoch']})")
            
#             del checkpoint
#             torch.cuda.empty_cache()

#             return start_epoch, max_accuracy
#         except:
#             del checkpoint
#             torch.cuda.empty_cache()
#             return 0, 0.

#     else:
#         logger.info(("=> no checkpoint found at '{}'".format(config.MODEL.RESUME)))
#         return 0, 0
    
def load_checkpoint(config, model, logger, use_pose_text_model = True):
    ## Load Vificlip Checkpoint
    if os.path.isfile(config.MODEL.RESUME): 
        logger.info(f"==============> Resuming form {config.MODEL.RESUME}....................")

        vificlip_checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')
        vificlip_weights = vificlip_checkpoint['model']

        # now remove the unwanted keys:
        if "module.prompt_learner.token_prefix" in vificlip_weights:
            del vificlip_weights["module.prompt_learner.token_prefix"]

        if "module.prompt_learner.token_suffix" in vificlip_weights:
            del vificlip_weights["module.prompt_learner.token_suffix"]

        if "module.prompt_learner.complete_text_embeddings" in vificlip_weights:
            del vificlip_weights["module.prompt_learner.complete_text_embeddings"]
        
        msg = model.load_state_dict(vificlip_weights, strict=False)

        ## check if module has to be removed from the keys?
        
        # new_state_dict = {}
        # for key in vificlip_weights:
        #     new_key = key.replace('module.', '')
        #     new_state_dict[new_key] = vificlip_weights[key]

        # msg = model.load_state_dict(new_state_dict, strict=False)
        logger.info(f"resume VIFICLIP model: {msg}")

    ## Load Hyperformer Checkpoint
    
    if config.MODEL.RESUME_POSE is not None and os.path.isfile(config.MODEL.RESUME_POSE): 
        logger.info(f"==============> Resuming from {config.MODEL.RESUME_POSE}....................")
        
        if config.MODEL.RESUME_POSE.endswith('.pt'):
            # msg = model.hyperformer_model.load_state_dict(config.MODEL.RESUME_POSE, strict=False)

            pose_text_weights = torch.load(config.MODEL.RESUME_POSE, map_location='cpu')
            # pose_text_weights = pose_text_checkpoint['model']
            new_state_dict_p1 = {}
            for key in pose_text_weights:
                new_key = 'module.hyperformer_model.' + key # .replace('module.hyperformer_model.', '')
                # if 'hyperformer_model' in key:
                new_state_dict_p1[new_key] = pose_text_weights[key]



            msg = model.load_state_dict(new_state_dict_p1, strict=False)
        
            logger.info(f"Load Hyperformer model: {msg}")
        elif config.MODEL.RESUME_POSE.endswith('.pth'):
            pose_text_checkpoint = torch.load(config.MODEL.RESUME_POSE, map_location='cpu')
            pose_text_weights = pose_text_checkpoint['model']

            if use_pose_text_model: # use text encoder weights from phase 1
                msg = model.load_state_dict(pose_text_weights, strict=False)

                # new_state_dict_p1 = {}
                # for key in pose_text_weights:
                #     new_key = key.replace('module.', '')
                #     new_state_dict_p1[new_key] = pose_text_weights[key]

                # msg = model.load_state_dict(new_state_dict_p1, strict=False)

            else: # use just Hyperformer weights from phase 1
                # msg = model.hyperformer_model.load_state_dict(pose_text_weights, strict=False)

                new_state_dict_p1 = {}
                for key in pose_text_weights:
                    # new_key = key.replace('module.hyperformer_model.', '')
                    if 'hyperformer_model' in key:
                        new_state_dict_p1[key] = pose_text_weights[key]

                msg = model.load_state_dict(new_state_dict_p1, strict=False)
                
            logger.info(f"resume Hyperformer model: {msg}")
    else:
        return 0, 0.0
    return 0, 0.0


def auto_resume_helper(output_dir):
    checkpoints = os.listdir(output_dir)
    checkpoints = [ckpt for ckpt in checkpoints if ckpt.endswith('pth')]
    print(f"All checkpoints founded in {output_dir}: {checkpoints}")
    if len(checkpoints) > 0:
        latest_checkpoint = max([os.path.join(output_dir, d) for d in checkpoints], key=os.path.getmtime)
        print(f"The latest checkpoint founded: {latest_checkpoint}")
        resume_file = latest_checkpoint
    else:
        resume_file = None
    return resume_file


def generate_text(data):
    text_aug = f"{{}}"
    classes = torch.cat([clip.tokenize(text_aug.format(c), context_length=77) for i, c in data.classes])

    return classes
