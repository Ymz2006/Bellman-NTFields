"""Offline speed supervision; inference receives configurations, never these labels."""
import numpy as np

COMPONENTS=['obstacle','self_collision','tilt_cup','tilt_pour']

def tilt_speed(angle_deg,limit_deg,transition_deg=10.,floor=.1):
    if not 0<floor<1 or not 0<transition_deg<=limit_deg:
        raise ValueError('Invalid tilt speed ramp')
    t=np.clip((np.asarray(angle_deg)-(limit_deg-transition_deg))/transition_deg,0.,1.)
    return (1.-(1.-floor)*t).astype(np.float32)

def position_weights(y,half_band=.1):
    if half_band<=0:raise ValueError('half_band must be positive')
    cup=np.clip((np.asarray(y)+half_band)/(2*half_band),0.,1.)
    return np.stack([cup,1.-cup],axis=-1).astype(np.float32)

def task_speeds(components,y,half_band=.1,floor=.1):
    """Keep physical components separate; mask inactive task speed, not cost-to-go."""
    physical=np.minimum(components[...,0],components[...,1])
    raw=np.minimum(physical[...,None],components[...,2:4])
    masked=raw.copy()
    masked[...,0]=np.where(y < -half_band,floor,masked[...,0])
    masked[...,1]=np.where(y > half_band,floor,masked[...,1])
    weights=position_weights(y,half_band)
    effective_or=masked.max(axis=-1)
    effective_weighted=(weights*masked).sum(axis=-1)
    return raw,masked,weights,effective_or,effective_weighted

def angles_from_signed_distance(distance,margin,label_limit,label_band):
    """Censored angle surrogate; not an exact angle beyond clipped-distance bounds."""
    return label_limit-label_band*np.asarray(distance)/margin

def legacy_components(obstacle,self_distance,cup_distance,pour_distance,margin=.05,offset=.005,self_scale=.5):
    d=np.stack([obstacle,self_distance*self_scale,cup_distance,pour_distance],axis=-1)
    return (np.clip(d,offset,margin)/margin).astype(np.float32)
