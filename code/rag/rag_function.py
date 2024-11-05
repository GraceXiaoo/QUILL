
def get_topk(vector,query,n,author):
    result_author =  vector.search_author(
        query=query,
        top_n=n,
        author = author
    )['documents']
    result = vector.search(
        query=query,
        top_n=n)['documents']
    if len(result_author[0])==5:
        return result_author
    else:
        new_result = []
        unique_list = []
        new_result.append(result_author[0]+result[0])
        for item in new_result:
            if item not in unique_list:
                unique_list.append(item)
        return [unique_list[0][:5]]
    
def retrieval(vector,old_context, n,author):
    try:
        old_context = old_context.replace('[Q]', '')
    except:
        pass
    topk_list = get_topk(vector,old_context, n,author)
    return topk_list
