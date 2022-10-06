  
from black import main
import torch
import numpy as np
import pdb
def aug_box_jitter(boxes, times=1, frac=0.06):
    def _aug_single(box):
        # random translate
        # TODO: random flip or something
        #box_scale = box[:, 2:4] - box[:, :2]
        box_scale = box[:, 2:4]
        box_scale = (
            box_scale.clamp(min=1)[:, None, :].expand(-1, 2, 2).reshape(-1, 4)
        )
        aug_scale = box_scale * frac  # [n,4]

        offset = (
            torch.randn(times, box.shape[0], 4, device=box.device)
            * aug_scale[None, ...]
        )
        new_box = box.clone()[None, ...].expand(times, box.shape[0], -1)
        return torch.cat(
            [new_box[:, :, :4].clone() + offset, new_box[:, :, 4:]], dim=-1
        )

    return [_aug_single(box) for box in boxes]

if __name__=='__main__':
    proposals = torch.randint(100, 1000, (20, 1, 4),device=0)
    print(proposals)
    print(proposals.shape)

    aug_proposals = aug_box_jitter(proposals, times=10, frac=0.5)
    aug_proposals = torch.stack(aug_proposals, dim=0)
    print(aug_proposals)
    print(aug_proposals.shape)
    pdb.set_trace()

