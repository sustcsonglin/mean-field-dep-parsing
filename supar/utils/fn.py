# -*- coding: utf-8 -*-

import unicodedata


def ispunct(token):
    return all(unicodedata.category(char).startswith('P')
               for char in token)


def isfullwidth(token):
    return all(unicodedata.east_asian_width(char) in ['W', 'F', 'A']
               for char in token)


def islatin(token):
    return all('LATIN' in unicodedata.name(char)
               for char in token)


def isdigit(token):
    return all('DIGIT' in unicodedata.name(char)
               for char in token)


def tohalfwidth(token):
    return unicodedata.normalize('NFKC', token)


def stripe(x, n, w, offset=(0, 0), dim=1):
    # r"""
    # Returns a diagonal stripe of the tensor.
    #
    # Args:
    #     x (~torch.Tensor): the input tensor with 2 or more dims.
    #     n (int): the length of the stripe.
    #     w (int): the width of the stripe.
    #     offset (tuple): the offset of the first two dims.
    #     dim (int): 1 if returns a horizontal stripe; 0 otherwise.
    #
    # Returns:
    #     a diagonal stripe of the tensor.
    # Examples:
    #     >>> x = torch.arange(25).view(5, 5)
    #     >>> x
    #     tensor([[ 0,  1,  2,  3,  4],
    #             [ 5,  6,  7,  8,  9],
    #             [10, 11, 12, 13, 14],
    #             [15, 16, 17, 18, 19],
    #             [20, 21, 22, 23, 24]])
    #     >>> stripe(x, 2, 3)
    #     tensor([[0, 1, 2],
    #             [6, 7, 8]])
    #     >>> stripe(x, 2, 3, (1, 1))
    #     tensor([[ 6,  7,  8],
    #             [12, 13, 14]])
    #     >>> stripe(x, 2, 3, (1, 1), 0)
    #     tensor([[ 6, 11, 16],
    #             [12, 17, 22]])
    # """

    x, seq_len = x.contiguous(), x.size(1)
    stride, numel = list(x.stride()), x[0, 0].numel()
    stride[0] = (seq_len + 1) * numel
    stride[1] = (1 if dim == 1 else seq_len) * numel
    return x.as_strided(size=(n, w, *x.shape[2:]),
                        stride=stride,
                        storage_offset=(offset[0]*seq_len+offset[1])*numel)


def pad(tensors, padding_value=0, total_length=None, padding_side='right'):
    size = [len(tensors)] + [max(tensor.size(i) for tensor in tensors)
                             for i in range(len(tensors[0].size()))]
    if total_length is not None:
        assert total_length >= size[1]
        size[1] = total_length
    out_tensor = tensors[0].data.new(*size).fill_(padding_value)
    for i, tensor in enumerate(tensors):
        out_tensor[i][[slice(-i, None) if padding_side == 'left' else slice(0, i) for i in tensor.size()]] = tensor
    return out_tensor


def diagonal_stripe(x, offset=1):
    r"""
    Returns a diagonal parallelogram stripe of the tensor.

    Args:
        x (~torch.Tensor): the input tensor with 3 or more dims.
        offset (int): which diagonal to consider. Default: 1.

    Returns:
        A diagonal parallelogram stripe of the tensor.

    Examples:
        >>> x = torch.arange(125).view(5, 5, 5)
        >>> diagonal_stripe(x)
        tensor([[ 5],
                [36],
                [67],
                [98]])
        >>> diagonal_stripe(x, 2)
        tensor([[10, 11],
                [41, 42],
                [72, 73]])
        >>> diagonal_stripe(x, -2)
        tensor([[ 50,  51],
                [ 81,  82],
                [112, 113]])
    """

    x = x.contiguous()
    seq_len, stride = x.size(1), list(x.stride())
    n, w, numel = seq_len - abs(offset), abs(offset), stride[2]
    return x.as_strided(size=(n, w, *x.shape[3:]),
                        stride=[((seq_len + 1) * x.size(2) + 1) * numel] + stride[2:],
                        storage_offset=offset*stride[1] if offset > 0 else abs(offset)*stride[0])


def expanded_stripe(x, n, w, offset=(0, 0)):
    r"""
    Returns an expanded parallelogram stripe of the tensor.

    Args:
        x (~torch.Tensor): the input tensor with 2 or more dims.
        n (int): the length of the stripe.
        w (int): the width of the stripe.
        offset (tuple): the offset of the first two dims.

    Returns:
        An expanded parallelogram stripe of the tensor.

    Examples:
        >>> x = torch.arange(25).view(5, 5)
        >>> x
        tensor([[ 0,  1,  2,  3,  4],
                [ 5,  6,  7,  8,  9],
                [10, 11, 12, 13, 14],
                [15, 16, 17, 18, 19],
                [20, 21, 22, 23, 24]])
        >>> expanded_stripe(x, 2, 3)
        tensor([[[ 0,  1,  2,  3,  4],
                [ 5,  6,  7,  8,  9],
                [10, 11, 12, 13, 14]],

                [[ 5,  6,  7,  8,  9],
                [10, 11, 12, 13, 14],
                [15, 16, 17, 18, 19]]])
        >>> expanded_stripe(x, 2, 3, (1, 1))
        tensor([[[ 5,  6,  7,  8,  9],
                [10, 11, 12, 13, 14],
                [15, 16, 17, 18, 19]],

                [[10, 11, 12, 13, 14],
                [15, 16, 17, 18, 19],
                [20, 21, 22, 23, 24]]])

    """
    x = x.contiguous()
    stride = list(x.stride())
    return x.as_strided(size=(n, w, *list(x.shape[1:])),
                        stride=stride[:1] + [stride[0]] + stride[1:],
                        storage_offset=(offset[1])*stride[0])


if __name__ == '__main__':
    import torch
    x = torch.arange(25).view(5, 5)
    print(stripe(x, 2, 3, (3, 1)))
