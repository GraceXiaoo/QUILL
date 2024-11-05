def mrr_score(target,rec_list):
    try:
        mrr=1/(rec_list.index(target)+1)
    except:
        mrr=0
    return mrr

def hits_at_k(target,rec_list,k):
    if k>len(rec_list):
        return 1
    try:
        target_index=rec_list[:k].index(target)
        return 1
    except:
        return 0