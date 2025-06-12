# slowfast/models/temporalclip_promptpool.py

from .build import MODEL_REGISTRY
from .temporalclip_video_model import TemporalClipVideo
from .prompt import Prompt
import json
import torch
from . import clip

@MODEL_REGISTRY.register()
class TemporalClipPromptPool(TemporalClipVideo):
    def __init__(self, cfg):
        super().__init__(cfg)
        
        self.prompt_enable = cfg.PROMPT_POOL.ENABLE # True
        self.pool_size = cfg.PROMPT_POOL.POOL_SIZE # 10
        self.top_k = cfg.PROMPT_POOL.TOP_K # 5
        self.prompt_length = cfg.PROMPT_POOL.PROMPT_LENGTH # 5
        self.embed_dim = 768 

        self.prompt = Prompt(
            length=self.prompt_length, embed_dim=self.embed_dim, embedding_key='cls',
            prompt_init='uniform', prompt_pool=True, prompt_key=True,
            pool_size=self.pool_size, top_k=self.top_k, batchwise_prompt=False,
            prompt_key_init='uniform'
        )  
                
    # parisa --------------------------------------------------
    def get_cls_feature(self, x=None):
        # shape of x(input) is (bz, channel, clip_len, h, w)
        # if len(x.shape) == 4:
        #     x = [x.unsqueeze(0)]

        assert len(x) == self.num_pathways
        x = x[0]
        if len(x.shape) == 4:
            # image input
            x = x.unsqueeze(2)
        
        bz, channel_dim, clip_len, h, w = x.shape
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(bz*clip_len, channel_dim, h, w)

        img_encode = self.model.encode_image(x)
        img_encode /= img_encode.norm(dim=-1, keepdim=True)

        img_encode = img_encode.reshape(bz, clip_len, -1)
        # Average across frames
        return img_encode.mean(dim=1) 
    # parisa --------------------------------------------------
    # we presume without label prompting
    def update_classifier(self, cfg, task_id, class_masks, flag = False):
        labels_dict = json.load(open(cfg.DATA.INDEX_LABEL_MAPPING_FILE, 'r'))
        
        # Update text_dict[0]
        if flag:
            for i in range(task_id+1):
                label_names = [labels_dict[str(idx)] for idx in class_masks[i]]
                new_tokens = torch.cat([clip.tokenize(name) for name in label_names])

                if i != 0:
                    self.text_dict[0] = torch.cat([self.text_dict[0], new_tokens], dim=0)
                else:
                    self.text_dict[0] = new_tokens        
        else:
            label_names = [labels_dict[str(idx)] for idx in class_masks[task_id]]
            new_tokens = torch.cat([clip.tokenize(name) for name in label_names])
            if task_id != 0:
                self.text_dict[0] = torch.cat([self.text_dict[0], new_tokens], dim=0)
            else:
                self.text_dict[0] = new_tokens            
        self.dynamic_classifier = self.achieve_csf_matrix(self.text_dict, self.model)  

    def forward(self, x=None, cls_features=None):

        if self.prompt_enable:
            res = self.prompt(prompt_mask=None, cls_features=cls_features)
            self.total_prompt_len = res['total_prompt_len']
            prompted_img_encode = res['prompted_embedding'] # [bz, prompt_len, dim]

        assert len(x) == self.num_pathways

        x = x[0]
        if len(x.shape) == 4:
            # image input
            x = x.unsqueeze(2)
        
        bz, channel_dim, clip_len, h, w = x.shape
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(bz*clip_len, channel_dim, h, w)
        if self.prompt_enable:
            img_encode = self.model.encode_image(x, prompted_img_encode)
        else:
            img_encode = self.model.encode_image(x, None)
        img_encode /= img_encode.norm(dim=-1, keepdim=True)
        # shape = (bz*clip_len, embed_dim)

        pred = self.model.logit_scale.exp() * img_encode @ self.dynamic_classifier.T
        if self.prompt_enable:
            res['logits'] = pred.reshape(bz, clip_len, -1).mean(1)
            return res  
        else:
            pred = pred.reshape(bz, clip_len, -1).mean(1)
            return pred

          
