
import torch
import numpy as np



def get_weight_prior(source_iters,num_classes=5):
    sample_count=torch.zeros((1,num_classes))
    print(sample_count)
    for iter in range(len(source_iters)):
        cur_sample_count=torch.zeros((1,num_classes))
        a = 0
        # for _,y in  iter:
        #     cur_sample_count+=torch.sum(y,dim=0)
        # sample_count=torch.cat([sample_count,cur_sample_count],dim=0)
        source_iters[iter] = np.array(source_iters[iter])

        for i in range(len(source_iters[iter])):
            a += len(source_iters[iter][i])
            for idx in range(5):
                cur_sample_count[0][idx] += len(np.argwhere(source_iters[iter][i]==idx))
        sample_count=torch.cat([sample_count,cur_sample_count],dim=0)

    sample_count=sample_count[1:][:]
    sample_count/=(torch.sum(sample_count,dim=0)+1)
    weight=sample_count.float()
    # weight=torch.softmax(sample_count.float(),dim=0)
    print(weight)
    return weight
