import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from experiments.c145_feasibility_reset.planar import wrap


def features(w,o):
    n=len(w.seeds);base=w.commands(o)[0];result=[];agents=w.p.shape[1]
    for i in range(agents):
        # Keep the legacy 34-D actor contract by exposing three cyclic peer
        # slots. P5-C uses an explicit active-member mask in its separate
        # controller; this legacy feature adapter is only for initialization.
        peers=(np.arange(agents)+i)%agents
        if agents > 3: peers=peers[:3]
        d=o['obstacles']-o['p'][:,i,None]
        clear=np.linalg.norm(d,axis=-1)-o['radii'];ix=np.arange(n);k=clear.argmin(1)
        target_delta=o['target']-o['p'][:,i];bearing=np.arctan2(target_delta[:,1],target_delta[:,0])
        yaw_error=wrap(bearing-w.yaw[:,i])
        x=np.concatenate([(o['p'][:,i]-o['target'])/12,(o['v'][:,i]-o['target_v'])/4,
            (o['z'][:,i,None]-3),o['vz'][:,i,None]/2,
            ((o['p'][:,peers]-o['p'][:,i,None])/12).reshape(n,-1),
            ((o['v'][:,peers]-o['v'][:,i,None])/4).reshape(n,-1),
            d[ix,k]/10,clear[ix,k,None]/5,o['target_v']/4,
            np.minimum(o['age'],20)[:,None]/5,np.sqrt(o['variance'])[:,None]/5,
            o['measured'][:,None],((o['searcher']==i)&(w.role_modes>0))[:,None],
            np.sin(yaw_error)[:,None],np.cos(yaw_error)[:,None],base[:,i]/4,
            np.full((n,1),w.time/w.task.duration),o['p'][:,i]/50],-1)
        assert x.shape==(n,34),x.shape;result.append(x)
    return np.clip(np.stack(result,1),-10,10).astype(np.float32)


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor=nn.Sequential(nn.Linear(34,64),nn.Tanh(),nn.Linear(64,64),nn.Tanh(),nn.Linear(64,8))
        self.critic=nn.Sequential(nn.Linear(102,128),nn.Tanh(),nn.Linear(128,64),nn.Tanh(),nn.Linear(64,1))
        for m in self.modules():
            if isinstance(m,nn.Linear):nn.init.orthogonal_(m.weight,np.sqrt(2));nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor[-1].weight,.01);nn.init.orthogonal_(self.critic[-1].weight,1.)
        with torch.no_grad():self.actor[-1].bias[0]=3.

    def forward(self,x):return Categorical(logits=self.actor(x)),self.critic(x.flatten(1)).squeeze(-1)


def optimize(net,opt,data,rng):
    reward=torch.as_tensor(data['reward'],dtype=torch.float32);value=torch.as_tensor(data['value'])
    advantage=torch.zeros_like(reward);last=torch.zeros(reward.shape[1])
    for t in reversed(range(len(reward))):
        nextv=value[t+1] if t+1<len(reward) else 0.
        last=reward[t]+.995*nextv-value[t]+.995*.95*last;advantage[t]=last
    returns=(advantage+value).flatten();adv=((advantage-advantage.mean())/(advantage.std()+1e-8)).flatten()
    x=torch.from_numpy(data['x'].reshape(-1,3,34));action=torch.from_numpy(data['action'].reshape(-1,3));old=torch.from_numpy(data['logp'].reshape(-1,3))
    rows=[]
    for _ in range(4):
        for idx in np.array_split(rng.permutation(len(x)),8):
            dist,val=net(x[idx]);lp=dist.log_prob(action[idx]);ratio=(lp-old[idx]).exp()
            actor=-torch.minimum(ratio*adv[idx,None],ratio.clamp(.8,1.2)*adv[idx,None]).mean()
            critic=.5*(val-returns[idx]).square().mean();entropy=dist.entropy().mean()
            loss=actor+.5*critic-.006*entropy;opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(net.parameters(),.5);opt.step()
            rows.append([actor.item(),critic.item(),entropy.item()])
    return dict(zip(['actor_loss','critic_loss','entropy'],np.mean(rows,0).tolist()))
