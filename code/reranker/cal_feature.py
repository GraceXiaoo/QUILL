from ..utils.utils import *
from ..app.app_compute import *

def cal_feature_ppl1(context,string,k=5):
    above=above_text(context)
    if len(string)>k:
        left_len=len(above+string[:k])
    else:
        left_len=len(above+string)
    ppl1=compute_ppl(left=context[:left_len],right=context[left_len:])
    if ppl1>20:
        ppl1=20
    return ppl1

def cal_feature_ppl2(context,string,k=5):
    above=above_text(context)
    left_len=len(above+string)
    ppl2=compute_ppl(left=context[:left_len],right=context[left_len:])
    if ppl2>20:
        ppl2=20
    return ppl2

def cal_feature_avg(context,string,k=5):
    ppl1 = cal_feature_ppl1(context, string, k)
    ppl2 = cal_feature_ppl2(context, string, k)
    return (ppl1 + ppl2) / 2

def cal_feature_ppl1_novelty(context,string,k=5):
    ppl1 = cal_feature_ppl1(context, string, k)
    novelty = get_novelty(string)
    return ppl1+novelty

def cal_feature_ppl2_novelty(context,string,k=5):
    ppl2 = cal_feature_ppl2(context, string, k)
    novelty = get_novelty(string)
    return ppl2+novelty

def cal_feature_avg_novelty(context,string,k=5):
    avg = cal_feature_avg(context, string, k)
    novelty = get_novelty(string)
    return avg+novelty