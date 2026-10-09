import argparse,time,json,hashlib
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .predictor import collect,predictions,metrics
from .common import snapshot,write_json


class EquivariantPredictor(nn.Module):
    def __init__(self,canonical=True):
        super().__init__();self.canonical=canonical;self.gru=nn.GRU(12,48,batch_first=True)
        self.mean=nn.Linear(48,4);self.logvar=nn.Linear(48,2);self.temperature=1.
        nn.init.zeros_(self.mean.weight);nn.init.zeros_(self.mean.bias)
        nn.init.zeros_(self.logvar.weight);nn.init.constant_(self.logvar.bias,-3.)

    def forward(self,x,hidden=None):
        if self.canonical:
            angle=torch.atan2(x[...,11],x[...,10]);cs=angle.cos();sn=angle.sin()
            # World -> causal velocity frame; non-vector features unchanged.
            z=x.clone()
            for a,b in [(0,1),(2,3),(10,11)]:
                z[...,a]=cs*x[...,a]+sn*x[...,b];z[...,b]=-sn*x[...,a]+cs*x[...,b]
        else:z=x
        h,hidden=self.gru(z,hidden);m=self.mean(h)
        if self.canonical:
            m=torch.stack([cs*m[...,0]-sn*m[...,1],sn*m[...,0]+cs*m[...,1],
                           cs*m[...,2]-sn*m[...,3],sn*m[...,2]+cs*m[...,3]],-1)
        return m,self.logvar(h).clamp(-8,4).repeat_interleave(2,-1),hidden


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--canonical',type=int,choices=[0,1],required=True);p.add_argument('--aux',type=float,required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--epochs',type=int,default=100);a=p.parse_args()
    torch.set_num_threads(1);torch.manual_seed(149);rng=np.random.default_rng(149)
    snapshot(a.out,dict(schema='c149_predictor_r2_v1',canonical=bool(a.canonical),aux=a.aux,epochs=a.epochs,seed=149,
        data=str(a.data),data_sha256={n:hashlib.sha256((a.data/f'{n}.npz').read_bytes()).hexdigest() for n in ['train','validation']},
        covariance='isotropic separately for position and velocity; invariant under rotations'))
    d=dict(np.load(a.data/'train.npz'));validation=dict(np.load(a.data/'validation.npz'))
    net=EquivariantPredictor(bool(a.canonical)).to(a.device);opt=torch.optim.Adam(net.parameters(),lr=8e-4)
    x=torch.from_numpy(d['x']).to(a.device);y=torch.from_numpy(d['y']).to(a.device)
    weight=torch.from_numpy(d['valid']*(1+3*(d['age']>.15))).to(a.device)
    best=np.inf;history=[];start=time.monotonic()
    for epoch in range(a.epochs):
        losses=[]
        for ids in np.array_split(rng.permutation(len(x)),24):
            mean,lv,_=net(x[ids]);sq=(y[ids]-mean).square()
            term=.5*(sq*torch.exp(-lv)+lv).mean(-1)+a.aux*sq.mean(-1)
            loss=(term*weight[ids]).sum()/weight[ids].sum()
            opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(net.parameters(),1);opt.step();losses.append(loss.item())
        m,v=predictions(net,validation,a.device);report=metrics(validation,m,v)
        score=report['lost']['position_rmse']+report['lost']['velocity_rmse']
        if score<best:
            best=score;torch.save(dict(state_dict=net.cpu().state_dict(),epoch=epoch+1,canonical=bool(a.canonical),aux=a.aux,validation=report),a.out/'best.pt');net.to(a.device)
        history.append(dict(epoch=epoch+1,loss=float(np.mean(losses)),validation=report))
        if (epoch+1)%20==0:print(json.dumps(dict(epoch=epoch+1,validation=report['lost'])),flush=True)
    write_json(a.out/'history.json',history);write_json(a.out/'completion.json',dict(epochs=a.epochs,training_episodes=512,wall_seconds=time.monotonic()-start))


def load_r2(path):
    ck=torch.load(path,map_location='cpu',weights_only=False);net=EquivariantPredictor(ck['canonical'])
    net.load_state_dict(ck['state_dict']);net.temperature=ck.get('temperature',1.);net.eval();return net

if __name__=='__main__':main()
