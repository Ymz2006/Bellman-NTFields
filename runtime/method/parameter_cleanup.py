"""Remove only proven-unused registered parameters; retain baseline RNG consumption.

The frozen constructor still runs before pruning. This deliberately preserves
historical random-number consumption; it is not a constructor speed optimization.
"""
import copy
import torch
from legacy_model import CurrentField

STAGES={
    1:('fusion_mlp.0.weight','fusion_mlp.0.bias','fusion_mlp.2.weight','fusion_mlp.2.bias'),
    2:('shared_encoder.0.weight','shared_encoder.0.bias'),
    3:('shared_gate.0.bias','shared_gate.1.bias'),
}

def removed_names(stage=3):
    return {name for i in range(1,stage+1) for name in STAGES[i]}

def prune(net,stage=3):
    if stage not in STAGES:raise ValueError('stage must be 1, 2 or 3')
    names=set(dict(net.named_parameters()))
    assert removed_names(stage)<=names
    if stage>=1:del net.fusion_mlp
    if stage>=2:net.shared_encoder[0]=torch.nn.Identity()
    if stage>=3:
        assert len(net.shared_gate)==2
        for gate in net.shared_gate:gate.register_parameter('bias',None)
    assert set(dict(net.named_parameters()))==names-removed_names(stage)
    return net

def remap_optimizer(optimizer_state,old_names,new_names):
    """Map Adam IDs by parameter name; refuse to drop accumulated optimizer state."""
    result=copy.deepcopy(optimizer_state)
    assert len(result['param_groups'])==1
    old_ids=result['param_groups'][0]['params'];assert len(old_ids)==len(old_names)
    ids=dict(zip(old_names,old_ids));removed=set(old_names)-set(new_names)
    assert not any(ids[n] in result['state'] for n in removed), 'Removed tensor has Adam state'
    result['state']={i:result['state'][ids[n]] for i,n in enumerate(new_names) if ids[n] in result['state']}
    result['param_groups'][0]['params']=list(range(len(new_names)))
    return result

def migrate_checkpoint(source,destination):
    ck=torch.load(source,map_location='cpu',weights_only=False)
    assert ck['repair_variant']=='samehead_no_bias' and not ck.get('parameter_cleanup_version')
    before_cpu=torch.get_rng_state().clone()
    net=CurrentField('cpu',6,ck['B_state_dict'],**ck['kwargs'])
    net.load_state_dict(ck['model_state_dict'],strict=True)
    old_names=list(dict(net.named_parameters()))
    prune(net);new_names=list(dict(net.named_parameters()))
    after=copy.deepcopy(ck)
    after['model_state_dict']={k:v for k,v in ck['model_state_dict'].items() if k not in removed_names()}
    net.load_state_dict(after['model_state_dict'],strict=True)
    after['optimizer_state_dict']=remap_optimizer(ck['optimizer_state_dict'],old_names,new_names)
    after['parameter_cleanup_version']=1
    after['removed_parameters']=sorted(removed_names())
    after['parameter_order']=new_names
    torch.save(after,destination)
    torch.set_rng_state(before_cpu)
    return after

def load_checkpoint(checkpoint,device='cuda:0'):
    ck=torch.load(checkpoint,map_location=device,weights_only=False)
    if ck.get('parameter_cleanup_version')!=1:raise ValueError('Expected the explicitly migrated compact checkpoint')
    net=CurrentField(device,6,ck['B_state_dict'],**ck['kwargs'])
    prune(net)
    net.load_state_dict(ck['model_state_dict'],strict=True);net.eval()
    assert list(dict(net.named_parameters()))==ck['parameter_order']
    q=torch.randn(4,12,device=device)*.1
    with torch.no_grad():
        xyz=net.position_fk(torch.cat([q[:,:6],q[:,6:]])*net.fk_range+net.fk_center)
        cost=net.out(q,ee_xy=xyz[:,:2],detach_input=False)[0]
        if not torch.isfinite(cost).all():raise ValueError('Nonfinite loaded cost')
    return net
