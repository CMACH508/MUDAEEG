import math
import numpy as np


def iterate_minibatches(inputs, targets, batch_size, shuffle=False):
    """
    Generate a generator that return a batch of inputs and targets.
    """

    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt]


def iterate_batch_seq_minibatches(inputs, targets, batch_size, seq_length):
    """
    Generate a generator that return a batch of sequences of inputs and targets.

    This function splits a sequence of inputs and targets into multiple sub-
    sequences equally. Then it further splits each sub-sequence into multiple
    chunks with the size of seq_length.
    """

    assert len(inputs) == len(targets)
    n_inputs = len(inputs)
    batch_len = n_inputs // batch_size

    epoch_size = batch_len // seq_length
    if epoch_size == 0:
        raise ValueError("epoch_size == 0, decrease batch_size or seq_length")

    seq_inputs = np.zeros((batch_size, batch_len) + inputs.shape[1:],
                          dtype=inputs.dtype)
    seq_targets = np.zeros((batch_size, batch_len) + targets.shape[1:],
                           dtype=targets.dtype)

    for i in range(batch_size):
        seq_inputs[i] = inputs[i*batch_len:(i+1)*batch_len]
        seq_targets[i] = targets[i*batch_len:(i+1)*batch_len]

    for i in range(epoch_size):
        x = seq_inputs[:, i*seq_length:(i+1)*seq_length]
        y = seq_targets[:, i*seq_length:(i+1)*seq_length]
        flatten_x = x.reshape((-1,) + inputs.shape[1:])
        flatten_y = y.reshape((-1,) + targets.shape[1:])
        yield flatten_x, flatten_y
import random

def iterate_batch_multiple_seq_minibatches_loop(inputs, targets, domain,batch_size, seq_length, shuffle_idx=None, augment_seq=False):
    """
    Generate a generator that return a batch of sequences of inputs and targets.

    This function randomly selects batches of multiple sequence. It then iterates
    through multiple sequence in parallel to generate a sequence of inputs and
    targets. It will append the input sequence with 0 and target with -1 when
    the lenght of each sequence is not equal.
    """

    assert len(inputs) == len(targets)
    n_inputs = len(inputs)

    if shuffle_idx is None:
        # No shuffle
        seq_idx = np.arange(n_inputs)
    else:
        # Shuffle subjects (get the shuffled indices from argument)
        seq_idx = shuffle_idx

    input_sample_shape = inputs[0].shape[1:]#输出为(3000,1,1)
    target_sample_shape = targets[0].shape[1:]#输出为()

    # Compute the number of maximum loops
    n_loops = int(math.ceil(len(seq_idx) / batch_size))#record数量/batchsize结果的上界

    # For each batch of subjects (size=batch_size)
    while True:
        for l in range(n_loops):
            start_idx = l*batch_size
            end_idx = (l+1)*batch_size
            seq_inputs = np.asarray(inputs,dtype = object)[seq_idx[start_idx:end_idx]]#取出start_idx开始包括它的record到end_idx前一个record
            seq_targets = np.asarray(targets,dtype = object)[seq_idx[start_idx:end_idx]]

            if augment_seq:
                # Data augmentation: multiple sequences
                # Randomly skip some epochs at the beginning -> generate multiple sequence
                max_skips = 5
                # print('aaaa')
                for s_idx in range(len(seq_inputs)):
                    n_skips = np.random.randint(max_skips)
                    seq_inputs[s_idx] = seq_inputs[s_idx][n_skips:]
                    seq_targets[s_idx] = seq_targets[s_idx][n_skips:]

            # find the maximum number of batch sequences,n_max_seq_inputs为batch_size个record中epoch最多的数量
            n_max_seq_inputs = -1
            for s_idx, s in enumerate(seq_inputs):#s_idx为i，s为seq_input[i]
                if len(s) > n_max_seq_inputs:
                    n_max_seq_inputs = len(s)

            n_batch_seqs = int(math.ceil(n_max_seq_inputs / seq_length))

            # For each batch sequence (size=seq_length)
            for b in range(n_batch_seqs):
                is_start_loop = True if b == 0 else False
                start_idx = b*seq_length
                end_idx = (b+1)*seq_length
                batch_inputs = np.zeros((batch_size, seq_length) + input_sample_shape, dtype=np.float32)#shape为(8, 20, 3000, 1, 1)
                batch_targets = np.zeros((batch_size, seq_length) + target_sample_shape, dtype=np.int)#shape为(8, 20)
                batch_weights = np.zeros((batch_size, seq_length), dtype=np.float32)#shape为(8, 20)
                batch_seq_len = np.zeros(batch_size, dtype=np.int)#shape为(8,)
                # print(batch_inputs.shape,batch_targets.shape,batch_weights.shape,batch_seq_len.shape)

                # For each subject
                for s_idx, s in enumerate(zip(seq_inputs, seq_targets)):
                    # (seq_len, sample_shape)
                    each_seq_inputs = s[0][start_idx:end_idx]#start_idx和end_idx相差20,第s_idx次的s[0]是第i个record的信号数值，s[i]是第i个record的标签
                    each_seq_targets = s[1][start_idx:end_idx]
                    batch_inputs[s_idx, :len(each_seq_inputs)] = each_seq_inputs#len(each_seq_inputs)=20,each_seq_inputs.shape=(20, 3000, 1, 1)，
                    # batch_inputs的第i个(20, 3000, 1, 1)表示第i个record的二十个epoch
                    batch_targets[s_idx, :len(each_seq_targets)] = each_seq_targets
                    batch_weights[s_idx, :len(each_seq_inputs)] = 1
                    batch_seq_len[s_idx] = len(each_seq_inputs)
                batch_x = batch_inputs.reshape((-1,1,3000))#batch_x.shape=(160, 3000, 1, 1)，其中前i*20~（i+1）*20个属于同一个record
                batch_y = batch_targets.reshape((-1,) + target_sample_shape)
                domains=np.zeros(batch_y.shape)
                #print(domains.shape)
                domains[:]=domain
                batch_weights = batch_weights.reshape(-1)
                yield batch_x, batch_y, domains,batch_weights, batch_seq_len, is_start_loop,n_loops
def iterate_batch_multiple_seq_minibatches(inputs, targets, domain,batch_size, seq_length, shuffle_idx=None, augment_seq=False):
    """
    Generate a generator that return a batch of sequences of inputs and targets.

    This function randomly selects batches of multiple sequence. It then iterates
    through multiple sequence in parallel to generate a sequence of inputs and
    targets. It will append the input sequence with 0 and target with -1 when
    the lenght of each sequence is not equal.
    """

    assert len(inputs) == len(targets)
    n_inputs = len(inputs)

    if shuffle_idx is None:
        # No shuffle
        seq_idx = np.arange(n_inputs)
    else:
        # Shuffle subjects (get the shuffled indices from argument)
        seq_idx = shuffle_idx

    input_sample_shape = inputs[0].shape[1:]
    target_sample_shape = targets[0].shape[1:]

    # Compute the number of maximum loops
    n_loops = int(math.ceil(len(seq_idx) / batch_size))
    #print(len(seq_idx), batch_size)
    # For each batch of subjects (size=batch_size)

    for l in range(n_loops):
        start_idx = l*batch_size
        end_idx = (l+1)*batch_size
        seq_inputs = np.asarray(inputs,dtype = object)[seq_idx[start_idx:end_idx]]
        seq_targets = np.asarray(targets,dtype = object)[seq_idx[start_idx:end_idx]]

        if augment_seq:
            # Data augmentation: multiple sequences
            # Randomly skip some epochs at the beginning -> generate multiple sequence
            max_skips = 5
            for s_idx in range(len(seq_inputs)):
                n_skips = np.random.randint(max_skips)
                seq_inputs[s_idx] = seq_inputs[s_idx][n_skips:]
                seq_targets[s_idx] = seq_targets[s_idx][n_skips:]

        # Determine the maximum number of batch sequences
        n_max_seq_inputs = -1
        for s_idx, s in enumerate(seq_inputs):
            if len(s) > n_max_seq_inputs:
                n_max_seq_inputs = len(s)

        n_batch_seqs = int(math.ceil(n_max_seq_inputs / seq_length))

        # For each batch sequence (size=seq_length)
        for b in range(n_batch_seqs):
            start_loop = True if b == 0 else False
            start_idx = b*seq_length
            end_idx = (b+1)*seq_length
            batch_inputs = np.zeros((batch_size, seq_length) + input_sample_shape, dtype=np.float32)
            batch_targets = np.zeros((batch_size, seq_length) + target_sample_shape, dtype=np.int)
            batch_weights = np.zeros((batch_size, seq_length), dtype=np.float32)
            batch_seq_len = np.zeros(batch_size, dtype=np.int)
            # For each subject
            for s_idx, s in enumerate(zip(seq_inputs, seq_targets)):
                # (seq_len, sample_shape)
                each_seq_inputs = s[0][start_idx:end_idx]
                each_seq_targets = s[1][start_idx:end_idx]
                batch_inputs[s_idx, :len(each_seq_inputs)] = each_seq_inputs
                batch_targets[s_idx, :len(each_seq_targets)] = each_seq_targets
                batch_weights[s_idx, :len(each_seq_inputs)] = 1
                batch_seq_len[s_idx] = len(each_seq_inputs)
            batch_x = batch_inputs.reshape((-1,1,3000))
            #print(batch_x.shape)
            batch_y = batch_targets.reshape((-1,) + target_sample_shape)
            domains=np.zeros(batch_y.shape)
            #print(domains.shape)
            domains[:]=domain
            batch_weights = batch_weights.reshape(-1)
            yield batch_x, batch_y, domains,batch_weights, batch_seq_len, start_loop,n_loops