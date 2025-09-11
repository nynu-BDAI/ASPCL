import torch
import torch.nn as nn
from torch.nn import functional as F

try:
    import torch.distributed.nn
    from torch import distributed as dist
    has_distributed = True
except ImportError:
    has_distributed = False

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

def gather_features(
        image_features,
        text_features,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
        use_horovod=False
):
    assert has_distributed, 'torch.distributed did not import correctly, please use a PyTorch version with support.'
    if use_horovod:
        assert hvd is not None, 'Please install horovod'
        if gather_with_grad:
            all_image_features = hvd.allgather(image_features)
            all_text_features = hvd.allgather(text_features)
        else:
            with torch.no_grad():
                all_image_features = hvd.allgather(image_features)
                all_text_features = hvd.allgather(text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features = list(all_image_features.chunk(world_size, dim=0))
                gathered_text_features = list(all_text_features.chunk(world_size, dim=0))
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
                all_image_features = torch.cat(gathered_image_features, dim=0)
                all_text_features = torch.cat(gathered_text_features, dim=0)
    else:
        # We gather tensors from all gpus
        if gather_with_grad:
            all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
            all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)
        else:
            gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
            gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
            dist.all_gather(gathered_image_features, image_features)
            dist.all_gather(gathered_text_features, text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
            all_image_features = torch.cat(gathered_image_features, dim=0)
            all_text_features = torch.cat(gathered_text_features, dim=0)

    return all_image_features, all_text_features
class ClipLoss(nn.Module):

    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        # calculated ground-truth and cache if enabled
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels

    def get_logits(self, image_features, text_features, logit_scale):
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features, text_features,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)

            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T
        
        return logits_per_image, logits_per_text

    def forward(self, image_features, text_features, logit_scale, output_dict=False):
        device = image_features.device
        logits_per_image, logits_per_text = self.get_logits(image_features, text_features, logit_scale)

        labels = self.get_ground_truth(device, logits_per_image.shape[0])

        total_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
        ) / 2

        return {"contrastive_loss": total_loss} if output_dict else total_loss

def gather_features_da(
        image_features,
        text_features,
        valid_caption_mask,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
        use_horovod=False
):
    assert has_distributed, 'torch.distributed did not import correctly, please use a PyTorch version with support.'
    if use_horovod:
        assert hvd is not None, 'Please install horovod'
        if gather_with_grad:
            all_image_features = hvd.allgather(image_features)
            all_text_features = hvd.allgather(text_features)
        else:
            with torch.no_grad():
                all_image_features = hvd.allgather(image_features)
                all_text_features = hvd.allgather(text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features = list(all_image_features.chunk(world_size, dim=0))
                gathered_text_features = list(all_text_features.chunk(world_size, dim=0))
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
                all_image_features = torch.cat(gathered_image_features, dim=0)
                all_text_features = torch.cat(gathered_text_features, dim=0)
    else:
        # We gather tensors from all gpus
        if gather_with_grad:
            all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
            all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)
            all_valid_caption_mask=torch.cat(torch.distributed.nn.all_gather(valid_caption_mask), dim=0)
        else:
            gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
            gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
            gathered_valid_caption_mask = [torch.zeros_like(valid_caption_mask) for _ in range(world_size)]
            dist.all_gather(gathered_image_features, image_features)
            dist.all_gather(gathered_text_features, text_features)
            dist.all_gather(gathered_valid_caption_mask, valid_caption_mask)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
                gathered_valid_caption_mask[rank] = valid_caption_mask
                
            all_image_features = torch.cat(gathered_image_features, dim=0)
            all_text_features = torch.cat(gathered_text_features, dim=0)
            all_valid_caption_mask = torch.cat(gathered_valid_caption_mask, dim=0)

    return all_image_features, all_text_features, all_valid_caption_mask


class Clip_DALoss(nn.Module):

    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
            cmr_loss=False,
            imc_loss=False,
            hardnegative=False,
            imc_loss_weight=0,
            cmr_loss_weight=0,
            threshold_type='mean',
    
            positive_margin_loss=False,
            positive_margin_loss_weight=0,
            analogy_loss=False,
            analogy_loss_weight=0,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        # cache state
        self.prev_num_logits = 0
        self.labels = {}
        self.cmr_loss = cmr_loss
        self.imc_loss = imc_loss
        self.imc_loss_weight = imc_loss_weight
        self.cmr_loss_weight = cmr_loss_weight
        self.threshold_type = threshold_type
        self.hardnegative = hardnegative
        
    
        self.positive_margin_loss = positive_margin_loss
        self.positive_margin_loss_weight = positive_margin_loss_weight
        if self.positive_margin_loss:
            self.alpha = nn.Parameter(torch.randn(1))
            
        self.analogy_loss = analogy_loss
        self.analogy_loss_weight = analogy_loss_weight
        

    def forward(self, image_features, text_features, valid_caption_mask, logit_scale, thresholds):

        device = image_features.device
        cmr_loss, imc_loss = 0.0, 0.0
        gt_similarity_diag = None 
        
        final_analogy_loss = torch.tensor(0.0, device=device)
        positive_loss = torch.tensor(0.0, device=device)

        if self.world_size > 1:
   
            all_image_features, all_text_features, all_valid_caption_mask = gather_features_da(
                image_features, text_features, valid_caption_mask,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)
            
            num_negatives = 5
            caption_types=torch.tensor(([1]*image_features.shape[0]+[2]*image_features.shape[0]*num_negatives)*self.world_size)
            
            gt_all_text_features=all_text_features[caption_types==1] # batch_size * word_size
            da_all_text_features=all_text_features[caption_types==2] # 4 * batch_size * word_size
            gt_len,feature_size=all_image_features.shape[0],all_image_features.shape[-1]

            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                if self.hardnegative:
                    all_text_features=torch.cat([gt_all_text_features,da_all_text_features])
                    logits_per_image = logit_scale * all_image_features @ all_text_features.T
                else:
                    logits_per_image = logit_scale * all_image_features @ gt_all_text_features.T

                logits_per_text = logit_scale * gt_all_text_features @ all_image_features.T

                if self.cmr_loss:
                    da_logits_per_image= logit_scale * (da_all_text_features.reshape(gt_len,-1,feature_size)@ all_image_features.unsqueeze(-1)).squeeze() * all_valid_caption_mask
                    cmr_loss,thresholds=self.get_cmr_loss(logits_per_image,da_logits_per_image,all_valid_caption_mask,thresholds)
                
                if self.imc_loss:
                    text_embedding_matrix=logit_scale * gt_all_text_features @ da_all_text_features.T
                    imc_loss+=self.get_imc_loss(logits_per_image,text_embedding_matrix)

        else:
      
            gt_len, feature_size = image_features.shape[0], image_features.shape[-1]
            
            num_texts_per_image = 10
            gt_text_features = text_features[::num_texts_per_image]
            
            indices_in_group = torch.arange(num_texts_per_image, device=device)
            da_mask_per_sample = (indices_in_group >= 1) & (indices_in_group <= 7)
            full_da_mask = da_mask_per_sample.repeat(gt_len)
            da_text_features = text_features[full_da_mask]
            
            base_logits_per_image = logit_scale * image_features @ gt_text_features.T
            logits_per_text = base_logits_per_image.T
            

            if self.hardnegative:
                
                all_text_features_for_loss = torch.cat([gt_text_features, da_text_features])
             
                logits_per_image = logit_scale * image_features @ all_text_features_for_loss.T
            else:
                logits_per_image = base_logits_per_image
            
      
            gt_similarity_diag = base_logits_per_image.diag()
            
      
            if self.cmr_loss:
                gt_logits_for_cmr = logits_per_image
                da_logits_per_image = logit_scale * (da_text_features.reshape(gt_len, -1, feature_size) @ image_features.unsqueeze(-1)).squeeze(-1)
                da_valid_caption_mask = valid_caption_mask[:, 0:7]

                cmr_loss, thresholds = self.get_cmr_loss(gt_logits_for_cmr, da_logits_per_image, da_valid_caption_mask, thresholds)
             
            gt_similarity_diag = base_logits_per_image.diag()
            
     
            if self.imc_loss:
            
                da_text_features_reshaped = da_text_features.reshape(gt_len, -1, feature_size)
                gt_text_features_expanded = gt_text_features.unsqueeze(1)
                semantic_diff_vectors = da_text_features_reshaped - gt_text_features_expanded
             
                final_perturbation = semantic_diff_vectors.detach()
    
                image_features_expanded = image_features.unsqueeze(1)
                image_negative_features = image_features_expanded + final_perturbation
                image_negative_features = F.normalize(image_negative_features, p=2, dim=-1)
                
          
                imc_loss += self.get_imc_loss(
                    logit_scale, image_features, image_negative_features, 
                    gt_text_features, da_text_features
                )
     

        num_logits = logits_per_image.shape[0]
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]

        total_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
        ) / 2

        if self.cmr_loss:
            total_loss += cmr_loss * self.cmr_loss_weight
        if self.imc_loss:
            total_loss += imc_loss * self.imc_loss_weight
            
     
        if self.positive_margin_loss and gt_similarity_diag is not None:
            unscaled_similarity = gt_similarity_diag / logit_scale
            self.alpha.data.clamp_(min=0.2) 
      
            positive_loss = F.relu(self.alpha - unscaled_similarity).mean()
            total_loss += self.positive_margin_loss_weight * positive_loss

  
        if self.analogy_loss and self.world_size <= 1: 
            hard_pos_features_0 = text_features[8::num_texts_per_image]
            hard_pos_features_1 = text_features[9::num_texts_per_image]

            valid_mask_hp0 = valid_caption_mask[:, 7].bool()
            valid_mask_hp1 = valid_caption_mask[:, 8].bool()

            text_loss_hp0 = torch.tensor(0.0, device=device)
            if valid_mask_hp0.any():
                valid_hp0_features = hard_pos_features_0[valid_mask_hp0]
                
                logits_hp0 = logit_scale * valid_hp0_features @ gt_text_features.T
         
                
                labels_hp0 = torch.where(valid_mask_hp0)[0].to(device)
                text_loss_hp0 = F.cross_entropy(logits_hp0, labels_hp0)

            text_loss_hp1 = torch.tensor(0.0, device=device)
            if valid_mask_hp1.any():
                valid_hp1_features = hard_pos_features_1[valid_mask_hp1]
                
                logits_hp1 = logit_scale * valid_hp1_features @ gt_text_features.T

                
                labels_hp1 = torch.where(valid_mask_hp1)[0].to(device)
                text_loss_hp1 = F.cross_entropy(logits_hp1, labels_hp1)
            
            text_analogy_loss = (text_loss_hp0 + text_loss_hp1) / 2.0
            
            semantic_offset_0 = hard_pos_features_0 - gt_text_features
            semantic_offset_1 = hard_pos_features_1 - gt_text_features
            
            hard_pos_image_features_0 = image_features + semantic_offset_0
            hard_pos_image_features_1 = image_features + semantic_offset_1
            
            hard_pos_image_features_0 = F.normalize(hard_pos_image_features_0, dim=-1)
            hard_pos_image_features_1 = F.normalize(hard_pos_image_features_1, dim=-1)

            image_loss_hp0 = torch.tensor(0.0, device=device)
            if valid_mask_hp0.any():
                valid_hp0_img_features = hard_pos_image_features_0[valid_mask_hp0]
                
                logits_img_hp0 = logit_scale * valid_hp0_img_features @ image_features.T
   
                
                labels_img_hp0 = torch.where(valid_mask_hp0)[0].to(device)
                image_loss_hp0 = F.cross_entropy(logits_img_hp0, labels_img_hp0)

            image_loss_hp1 = torch.tensor(0.0, device=device)
            if valid_mask_hp1.any():
                valid_hp1_img_features = hard_pos_image_features_1[valid_mask_hp1]
                
                logits_img_hp1 = logit_scale * valid_hp1_img_features @ image_features.T
        
                
                labels_img_hp1 = torch.where(valid_mask_hp1)[0].to(device)
                
                image_loss_hp1 = F.cross_entropy(logits_img_hp1, labels_img_hp1)
                
            image_analogy_loss = (image_loss_hp0 + image_loss_hp1) / 2.0
            final_analogy_loss = text_analogy_loss + 0.0001 * image_analogy_loss
            total_loss += self.analogy_loss_weight * final_analogy_loss
        return total_loss, thresholds, cmr_loss, imc_loss, final_analogy_loss, positive_loss
        
    def get_cmr_loss(self, gt_logits_per_image: torch.Tensor, da_logits_per_image: torch.Tensor, valid_caption_mask, thresholds: torch.Tensor) -> torch.Tensor:
       
        gt_similarity = gt_logits_per_image.diag().reshape(-1, 1).expand_as(da_logits_per_image)
        cmr_loss = nn.functional.relu((thresholds + da_logits_per_image - gt_similarity)) * valid_caption_mask
        
        # updating thresholds
        if self.threshold_type == 'mean':
            mask = valid_caption_mask.bool()
            valid_counts = mask.sum(dim=0)
            average_similarity_for_types = torch.where(
                valid_counts > 0,
                (da_logits_per_image * mask).sum(dim=0) / valid_counts,
                torch.zeros_like(valid_counts, dtype=da_logits_per_image.dtype)
            )
            thresholds = (gt_similarity.mean(0) - average_similarity_for_types).expand_as(gt_similarity)
            thresholds = thresholds.detach()
        elif self.threshold_type == 'max':
            thresholds, max_indices = (gt_similarity * valid_caption_mask - da_logits_per_image).max(0)
            thresholds = thresholds.expand_as(gt_similarity) / 5
            thresholds = thresholds.detach()
        return cmr_loss.mean(), thresholds

    def get_imc_loss(
        self,
        logit_scale: torch.Tensor,
        image_positive_features: torch.Tensor,
        image_negative_features: torch.Tensor,
        text_positive_features: torch.Tensor,
        text_negative_features: torch.Tensor,
    ):
 
        device = image_positive_features.device
        text_logits = logit_scale * text_positive_features @ text_negative_features.T
        image_neg_flat = image_negative_features.reshape(-1, image_negative_features.shape[-1])
        image_logits = logit_scale * image_positive_features @ image_neg_flat.T
        labels = torch.zeros(text_logits.shape[0], device=device, dtype=torch.long)
        loss_text_side = F.cross_entropy(text_logits, labels)
        loss_image_side = F.cross_entropy(image_logits, labels)
        return loss_text_side + 0.0001 * loss_image_side