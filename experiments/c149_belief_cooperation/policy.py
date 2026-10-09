"""Shared local actor, ideal message exchange, centralized legal-information value."""
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


def features(w,o):
    n=len(o['p']);contexts=[];nodes=[]
    nominal=w.nominal(o)
    for i in range(3):
        nearest_delta=o['obstacles']-o['p'][:,i,None]
        clear=np.linalg.norm(nearest_delta,axis=-1)-o['radii']
        nearest=clear.argmin(1);ix=np.arange(n)
        peers=(np.arange(3)+i)%3
        contexts.append(np.concatenate([
            (o['p'][:,i]-o['target'])/10,(o['v'][:,i]-o['target_v'])/4,o['p'][:,i]/50,
            ((o['p'][:,peers]-o['p'][:,i,None])/10).reshape(n,-1),
            ((o['v'][:,peers]-o['v'][:,i,None])/4).reshape(n,-1),
            nearest_delta[ix,nearest]/10,clear[ix,nearest,None]/10,
            np.full((n,1),w.time/30),nominal[:,i]/4],-1))
        nodes.append(np.concatenate([(o['local_p']-o['p'][:,i,None])/10,o['local_v']/4,
            np.minimum(o['local_age'],10)[...,None]/3,
            np.log(np.maximum(o['local_var'],.001))[...,None]/4,
            o['local_mask'][...,None],o['local_valid'][...,None]],-1))
    context=np.clip(np.stack(contexts,1),-10,10).astype(np.float32)
    node=np.clip(np.stack(nodes,1),-10,10).astype(np.float32)
    assert context.shape==(n,3,24) and node.shape==(n,3,3,8)
    return context,node


class Policy(nn.Module):
    def __init__(self,confidence=False):
        super().__init__();self.confidence=confidence
        self.query=nn.Linear(24,32);self.key=nn.Linear(8,32);self.message=nn.Linear(8,32)
        self.actor=nn.Sequential(nn.Linear(56,64),nn.Tanh(),nn.Linear(64,5))
        self.critic=nn.Sequential(nn.Linear(144,128),nn.Tanh(),nn.Linear(128,64),nn.Tanh(),nn.Linear(64,1))
        for m in self.modules():
            if isinstance(m,nn.Linear):nn.init.orthogonal_(m.weight,np.sqrt(2));nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor[-1].weight,.01);nn.init.orthogonal_(self.critic[-1].weight,1.)
        with torch.no_grad():self.actor[-1].bias[0]=2.

    def forward(self,context,nodes):
        q=torch.tanh(self.query(context));k=torch.tanh(self.key(nodes));v=torch.tanh(self.message(nodes))
        logits=(q.unsqueeze(-2)*k).sum(-1)/np.sqrt(32)
        if self.confidence:logits=logits-4*nodes[...,5]
        logits=logits.masked_fill(nodes[...,7]<.5,-1e9)
        attention=torch.softmax(logits,-1);pooled=(attention.unsqueeze(-1)*v).sum(-2)
        distribution=Categorical(logits=self.actor(torch.cat([context,pooled],-1)))
        central=torch.cat([context.flatten(1),nodes.flatten(1)],-1)
        return distribution,self.critic(central).squeeze(-1),attention
