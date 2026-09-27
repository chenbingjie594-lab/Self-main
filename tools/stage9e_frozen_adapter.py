"""Numerically repaired Stage9A adapter; architecture and parameterization unchanged."""
from __future__ import annotations
from pathlib import Path
import torch
import torch.nn.functional as F
from stage9a_frozen_adapter import FrozenResidualAdapter,timestep_embedding

class RepairedFrozenResidualAdapter(FrozenResidualAdapter):
    rms_epsilon=1e-6
    def _residual(self,hidden,index):
        if self._mask is None:raise RuntimeError("Stage9E condition was not set")
        size=hidden.shape[-2:]
        mask=F.interpolate(self._mask.float(),size=size,mode="nearest")
        context=F.interpolate(self._context.float(),size=size,mode="bilinear",align_corners=False)
        feature=self.mask_encoder(mask)+self.context_encoder(context)
        time=self.time_mlp(timestep_embedding(self._timesteps,self.hidden)).to(feature.dtype)
        feature=F.silu(feature+time[:,:,None,None])*mask
        residual=self.projections[index](feature).to(hidden.dtype)
        eps=self.rms_epsilon
        h_rms=torch.sqrt(hidden.float().square().mean((1,2,3),keepdim=True)+eps**2)
        r_rms=torch.sqrt(residual.float().square().mean((1,2,3),keepdim=True)+eps**2)
        scale=(self.max_rms_ratio*h_rms/r_rms).clamp(max=1.0).to(residual.dtype)
        return residual*scale

    @classmethod
    def load(cls,path,device="cpu"):
        payload=torch.load(path,map_location=device,weights_only=True)
        obj=cls(**payload["metadata"]["architecture"]);obj.load_state_dict(payload["state_dict"])
        return obj,payload["metadata"]

