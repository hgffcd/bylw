"""Gaussian GRU residual over a classical CT prior; supervised truth is labels only."""
import argparse,time,json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .world import LocalWorld,Conditions
from .common import snapshot,write_json


class Predictor(nn.Module):
    def __init__(self):
        super().__init__();self.gru=nn.GRU(12,48,batch_first=True)
        self.mean=nn.Linear(48,4);self.logvar=nn.Linear(48,4)
        nn.init.zeros_(self.mean.weight);nn.init.zeros_(self.mean.bias)
        nn.init.zeros_(self.logvar.weight);nn.init.constant_(self.logvar.bias,-3.)
        self.temperature=1.

    def forward(self,x,hidden=None):
        h,hidden=self.gru(x,hidden)
        return self.mean(h),self.logvar(h).clamp(-8,4),hidden


def collect(base,n,out):
    blocks=[]
    batch=min(64,n//4)
    for begin in range(0,n,batch):
        seeds=np.arange(base+begin,base+min(begin+batch,n))
        loss=[.8,1.2,2.,3.][(begin//batch)%4]
        w=LocalWorld(seeds,Conditions(loss=loss));features=[];labels=[];ages=[];valid=[];variance=[]
        for _ in range(300):
            o=w.observe();features.append(o['history_feature']);ages.append(o['local_age']);valid.append(o['local_valid'])
            labels.append(np.concatenate([(w.target[:,None]-o['prior_p'])/5,(w.tv[:,None]-o['prior_v'])/4],-1))
            variance.append(o['prior_var']);w.step(o)
        # Independent local-agent sequences; no episode/sensor boundary crossing.
        blocks.append(dict(x=np.array(features).transpose(1,2,0,3).reshape(-1,300,12),
            y=np.array(labels).transpose(1,2,0,3).reshape(-1,300,4),
            age=np.array(ages).transpose(1,2,0).reshape(-1,300),
            valid=np.array(valid).transpose(1,2,0).reshape(-1,300),
            variance=np.array(variance).transpose(1,2,0).reshape(-1,300)))
    d={k:np.concatenate([b[k] for b in blocks]).astype(bool if k=='valid' else np.float32) for k in blocks[0]}
    np.savez_compressed(out,**d,seeds=np.arange(base,base+n),loss_durations=np.array([.8,1.2,2.,3.]))
    return d


def predictions(net,d,device='cpu'):
    ms=[];vs=[]
    with torch.no_grad():
        for start in range(0,len(d['x']),64):
            m,lv,_=net(torch.from_numpy(d['x'][start:start+64]).to(device))
            ms.append(m.cpu().numpy());vs.append(lv.exp().cpu().numpy())
    return np.concatenate(ms),np.concatenate(vs)


def metrics(d,mean,var,temperature=1.):
    err=(d['y']-mean)*np.array([5,5,4,4]);base=d['y']*np.array([5,5,4,4])
    out={}
    for name,mask in [('all',d['valid']),('lost',d['valid']&(d['age']>.15)),
                       ('long_loss',d['valid']&(d['age']>=1.5))]:
        e=err[mask];b=base[mask];v=var[mask,:2]*25*temperature
        if not len(e):continue
        radius_stat=(e[:,:2]**2/np.maximum(v,1e-8)).sum(-1)
        out[name]=dict(samples=int(mask.sum()),position_rmse=float(np.sqrt((e[:,:2]**2).sum(-1).mean())),
            velocity_rmse=float(np.sqrt((e[:,2:]**2).sum(-1).mean())),
            ct_position_rmse=float(np.sqrt((b[:,:2]**2).sum(-1).mean())),
            ct_velocity_rmse=float(np.sqrt((b[:,2:]**2).sum(-1).mean())),
            joint_ellipse95_coverage=float((radius_stat<=5.991464547).mean()),
            ellipse95_area_mean=float((np.pi*5.991464547*np.sqrt(v.prod(-1))).mean()))
    return out


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--epochs',type=int,default=40)
    p.add_argument('--device',default='cuda:0');a=p.parse_args();torch.set_num_threads(1);torch.manual_seed(149)
    np.random.seed(149);rng=np.random.default_rng(149)
    snapshot(a.out,dict(schema='c149_local_gru_v1',seed=149,epochs=a.epochs,device=a.device,
        train=[14910000,512],validation=[14920000,128],calibration=[14921000,128],test=[14922000,256],
        labels='simulator truth only for offline supervised residual targets',sequence='300 ticks, independently reset each episode and sensor'))
    start=time.monotonic();datasets={}
    for name,base,n in [('train',14910000,512),('validation',14920000,128),('calibration',14921000,128),('test',14922000,256)]:
        datasets[name]=collect(base,n,a.out/f'{name}.npz');print('collected',name,flush=True)
    net=Predictor().to(a.device);opt=torch.optim.Adam(net.parameters(),lr=8e-4);history=[];best=np.inf
    d=datasets['train'];x=torch.from_numpy(d['x']).to(a.device);y=torch.from_numpy(d['y']).to(a.device)
    weight=torch.from_numpy(d['valid']*(1+3*(d['age']>.15))).to(a.device)
    for epoch in range(a.epochs):
        losses=[]
        for ids in np.array_split(rng.permutation(len(x)),24):
            mean,lv,_=net(x[ids]);loss=(.5*((y[ids]-mean).square()*torch.exp(-lv)+lv).mean(-1)*weight[ids]).sum()/weight[ids].sum()
            opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(net.parameters(),1);opt.step();losses.append(loss.item())
        m,v=predictions(net,datasets['validation'],a.device);report=metrics(datasets['validation'],m,v)
        score=report['lost']['position_rmse']+report['lost']['velocity_rmse']
        if score<best:
            best=score;torch.save(dict(state_dict=net.cpu().state_dict(),epoch=epoch+1,validation=report),a.out/'best.pt');net.to(a.device)
        history.append(dict(epoch=epoch+1,nll=float(np.mean(losses)),validation=report))
        if (epoch+1)%5==0:print(json.dumps(history[-1]),flush=True)
    ck=torch.load(a.out/'best.pt',map_location='cpu',weights_only=False);net.cpu().load_state_dict(ck['state_dict'])
    cal=datasets['calibration'];m,v=predictions(net.cpu(),cal);mask=cal['valid']&(cal['age']>.15)
    statistic=(((cal['y'][...,:2]-m[...,:2])**2/v[...,:2]).sum(-1))[mask]
    temperature=float(np.quantile(statistic,.95)/5.991464547)
    ck['temperature']=temperature;torch.save(ck,a.out/'calibrated.pt')
    reports={}
    for name in ['validation','calibration','test']:
        m,v=predictions(net,datasets[name]);reports[name]=metrics(datasets[name],m,v,temperature)
    val=reports['validation']['lost'];test=reports['test']['lost']
    # Predeclared validation gate; test is reporting, not configuration selection.
    passed=(val['position_rmse']<.9*val['ct_position_rmse'] and val['velocity_rmse']<.9*val['ct_velocity_rmse'])
    write_json(a.out/'result.json',dict(best_epoch=ck['epoch'],temperature=temperature,validation_gate_pass=bool(passed),
        gate='validation lost position and velocity RMSE each improve by >=10%; test not used for model selection',
        metrics=reports,wall_seconds=time.monotonic()-start,training_episodes=512,
        note='one frozen predictor seed; three later PPO seeds are not three end-to-end predictor seeds'))
    write_json(a.out/'history.json',history);print(json.dumps(dict(completed=True,validation_gate_pass=bool(passed),test=test)),flush=True)


def load_predictor(path):
    net=Predictor();ck=torch.load(path,map_location='cpu',weights_only=False)
    net.load_state_dict(ck['state_dict']);net.temperature=ck.get('temperature',1.);net.eval();return net


if __name__=='__main__':main()
