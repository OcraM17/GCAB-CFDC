import copy

import torch
from torch import nn
from .dytox import DyTox
from timm.models.layers import trunc_normal_
import continual.utils as cutils
from .dytox import ContinualClassifier


class GCAB(DyTox):

    def __init__(
            self,
            transformer,
            nb_classes,
            individual_classifier='',
            head_div=False,
            head_div_mode='',
            joint_tokens=False,
            num_tasks=10,
            thres_cosh=50,
            thres_emb=6,
            smax=800,
            lambda_gcab=0.05,
            lambda_pfr=0.001

    ):
        super().__init__(transformer,
                         nb_classes,
                         individual_classifier,
                         head_div,
                         head_div_mode,
                         joint_tokens)

        self.embs_0 = nn.Embedding(num_tasks, 384).cuda()
        self.embs_2 = nn.Embedding(num_tasks, 1536).cuda()
        self.projectors = nn.ModuleList([])
        self.gate = nn.Sigmoid()

        self.thres_cosh = thres_cosh
        self.thres_emb = thres_emb
        self.old_model = None
        self.smax = smax
        self.lambda_gcab = lambda_gcab
        self.lambda_pfr = lambda_pfr
        self.mask_pre = None
        self.head_div = None
        self.use_head_div = False
        self.tabs = transformer.blocks[transformer.local_up_to_layer:]
        self.class_tokens = nn.ParameterList([transformer.cls_token])

        if self.individual_classifier != '':
            in_dim, out_dim = self._get_ind_clf_dim()
            self.head = nn.ModuleList([
                ContinualClassifier(in_dim, out_dim).cuda()
            ])
        else:
            self.head = ContinualClassifier(
                self.embed_dim * len(self.class_tokens), sum(self.nb_classes_per_task)
            ).cuda()

    def get_view_for(self, n, masks):
        gc0, gc2 = masks
        if n == 'tabs.0.attn.proj.weight':
            return torch.matmul(gc0.view(-1, 1), gc0)
        if n == 'tabs.0.attn.proj.bias':
            return gc0.view(-1)
        if n == 'tabs.0.attn.k.weight':
            return torch.matmul(gc0.view(-1, 1), gc0)
        if n == 'tabs.0.attn.v.weight':
            return torch.matmul(gc0.view(-1, 1), gc0)
        if n == 'tabs.0.attn.q.weight':
            return torch.matmul(gc0.view(-1, 1), gc0)
        if n == 'tabs.0.mlp.fc1.weight':
            return torch.matmul(gc2.view(-1, 1), gc0)
        if n == 'tabs.0.mlp.fc1.bias':
            return gc2.view(-1)
        if n == 'tabs.0.mlp.fc2.weight':
            return torch.matmul(gc0.view(-1, 1), gc2)
        if n == 'tabs.0.mlp.fc2.bias':
            return gc0.view(-1)

    def _get_ind_clf_dim(self):
        if self.individual_classifier == '1-1':
            in_dim = self.embed_dim
            out_dim = self.nb_classes_per_task[-1]
        elif self.individual_classifier == '1-n':
            in_dim = self.embed_dim
            out_dim = sum(self.nb_classes_per_task)
        elif self.individual_classifier == 'n-n':
            in_dim = len(self.class_tokens) * self.embed_dim
            out_dim = sum(self.nb_classes_per_task)
        elif self.individual_classifier == 'n-1':
            in_dim = len(self.class_tokens) * self.embed_dim
            out_dim = self.nb_classes_per_task[-1]
        else:
            raise NotImplementedError(f'Unknown ind classifier {self.individual_classifier}')
        return in_dim, out_dim

    def freeze(self, names):
        """Choose what to freeze depending on the name of the module."""
        requires_grad = False
        cutils.freeze_parameters(self, requires_grad=not requires_grad)
        self.train()

        for name in names:
            if name == 'all':
                self.eval()
                return cutils.freeze_parameters(self)
            elif name == 'old_task_tokens':
                cutils.freeze_parameters(self.class_tokens[:-1], requires_grad=requires_grad)
            elif name == 'task_tokens':
                cutils.freeze_parameters(self.class_tokens, requires_grad=requires_grad)
            elif name == 'sab':
                self.sabs.eval()
                cutils.freeze_parameters(self.patch_embed, requires_grad=requires_grad)
                cutils.freeze_parameters(self.pos_embed, requires_grad=requires_grad)
                cutils.freeze_parameters(self.sabs, requires_grad=requires_grad)
            elif name == 'tab':
                self.tabs.eval()
                cutils.freeze_parameters(self.tabs, requires_grad=requires_grad)
            elif name == 'old_heads':
                self.head[:-1].eval()
                cutils.freeze_parameters(self.head[:-1], requires_grad=requires_grad)
            elif name == 'heads':
                self.head.eval()
                cutils.freeze_parameters(self.head, requires_grad=requires_grad)
            else:
                raise NotImplementedError(f'Unknown name={name}.')

    def add_model(self, nb_new_classes):
        self.nb_classes_per_task.append(nb_new_classes)

        new_task_token = copy.deepcopy(self.class_tokens[-1])
        trunc_normal_(new_task_token, std=.02)
        self.class_tokens.append(new_task_token)
        if self.individual_classifier != '':
            in_dim, out_dim = self._get_ind_clf_dim()
            self.head.append(
                ContinualClassifier(in_dim, out_dim).cuda()
            )
        else:
            self.head = ContinualClassifier(
                self.embed_dim * len(self.class_tokens), sum(self.nb_classes_per_task)
            ).cuda()
        # ----------------------------------------------------------------------

    def forward_features(self, x, s, inference):
        B = x.shape[0]

        x = self.patch_embed(x)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        s_e, s_a, s_v = [], [], []
        for blk in self.sabs:
            x, attn, v = blk(x)
            s_e.append(x)
            s_a.append(attn)
            s_v.append(v)

        tokens = []
        attentions = []
        mask_heads = None
        feats = x.clone()
        for c, class_token in enumerate(self.class_tokens):
            class_token = class_token.expand(B, -1, -1)

            ca_blocks = self.tabs

            for blk in ca_blocks:
                x_act = x.clone()
                if len(self.class_tokens) > 1 and inference is not None:
                    x_act = torch.permute(x_act.unsqueeze(2), (0, 3, 2, 1))
                    for ind in range(len(self.projectors) - 1, c - 1, -1):
                        x_act = self.projectors[ind](x_act)
                    x_act = torch.permute(x_act.squeeze(2), (0, 2, 1))
                class_token, attn, v, masks = blk(torch.cat((class_token, x_act), dim=1), s, self,
                                                  mask_heads=mask_heads,
                                                  c=c + 1)

            attentions.append(attn)
            tokens.append(class_token[:, 0])

        self._class_tokens = tokens
        return tokens, tokens[-1], attentions, masks, feats

    def forward_classifier(self, tokens, last_token):

        if self.individual_classifier != '':
            logits = []

            for i, head in enumerate(self.head):
                if self.individual_classifier in ('1-n', '1-1'):
                    logits.append(head(tokens[i]))
                else:  # n-1, n-n
                    logits.append(head(torch.cat(tokens[:i + 1], dim=1)))

            if self.individual_classifier in ('1-1', 'n-1'):
                logits = torch.cat(logits, dim=1)
            else:  # 1-n, n-n
                final_logits = torch.zeros_like(logits[-1])
                for i in range(len(logits)):
                    final_logits[:, :logits[i].shape[1]] += logits[i]

                for i, c in enumerate(self.nb_classes_per_task):
                    final_logits[:, :c] /= len(self.nb_classes_per_task) - i

                logits = final_logits
        elif isinstance(tokens, torch.Tensor):
            logits = self.head(tokens)
        else:
            logits = self.head(torch.cat(tokens, dim=1))

        return logits

    def forward(self, x, s, inference):
        tokens, last_token, _, masks, feats = self.forward_features(x, s, inference)
        return self.forward_classifier(tokens, last_token), masks, feats

    def param_groups(self):
        return {
            'all': self.parameters(),
            'old_task_tokens': self.class_tokens[:-1],
            'task_tokens': self.class_tokens.parameters(),
            'new_task_tokens': [self.class_tokens[-1]],
            'sa': self.sabs.parameters(),
            'patch': self.patch_embed.parameters(),
            'pos': [self.pos_embed],
            'ca': self.tabs.parameters(),
            'old_heads': self.head[:-self.nb_classes_per_task[-1]].parameters() \
                if self.individual_classifier else \
                self.head.parameters(),
            'new_head': self.head[-1].parameters() if self.individual_classifier else self.head.parameters(),
            'head': self.head.parameters(),
        }
