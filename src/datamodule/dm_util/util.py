import unicodedata

punct_set = '.' '``' "''" ':' ','
import re


# https://github.com/DoodleJZ/HPSG-Neural-Parser/blob/cdcffa78945359e14063326cadd93fd4c509c585/src_joint/dep_eval.py
def is_uni_punctuation(word):
    match = re.match("^[^\w\s]+$]", word, flags=re.UNICODE)
    return match is not None

def is_punctuation(word, pos, punct_set=punct_set):
    if punct_set is None:
        return is_uni_punctuation(word)
    else:
        return pos in punct_set or pos == 'PU' # for chinese

def get_path(path):
    return path

def get_path_debug(path):
    return path + ".debug"

def clean_number(w):
    new_w = re.sub('[0-9]{1,}([,.]?[0-9]*)*', '0', w)
    return new_w

def clean_word(words):
    PTB_UNESCAPE_MAPPING = {
        "«": '"',
        "»": '"',
        "‘": "'",
        "’": "'",
        "“": '"',
        "”": '"',
        "„": '"',
        "‹": "'",
        "›": "'",
        "\u2013": "--",  # en dash
        "\u2014": "--",  # em dash
    }
    cleaned_words = []
    for word in words:
        word = PTB_UNESCAPE_MAPPING.get(word, word)
        word = word.replace("\\/", "/").replace("\\*", "*")
        # Mid-token punctuation occurs in biomedical text
        word = word.replace("-LSB-", "[").replace("-RSB-", "]")
        word = word.replace("-LRB-", "(").replace("-RRB-", ")")
        word = word.replace("-LCB-", "{").replace("-RCB-", "}")
        word = word.replace("``", '"').replace("`", "'").replace("''", '"')
        word = clean_number(word)
        cleaned_words.append(word)
    return cleaned_words




def find_dep_boundary(heads):
    left_bd = [i for i in range(len(heads))]
    right_bd = [i + 1  for i in range(len(heads))]

    for child_idx, head_idx in enumerate(heads):
        if head_idx > 0:
            if left_bd[child_idx] < left_bd[head_idx - 1]:
                left_bd[head_idx - 1] = left_bd[child_idx]

            elif child_idx > right_bd[head_idx - 1] - 1:
                right_bd[head_idx - 1] = child_idx + 1
                while head_idx != 0:
                    if heads[head_idx-1] > 0 and  child_idx + 1 > right_bd[ heads[head_idx-1] - 1] :
                        right_bd[ heads[head_idx-1] - 1]  =  child_idx + 1
                        head_idx = heads[head_idx-1]
                    else:
                        break

    # (head_word_idx, left_bd_idx, right_bd_idx)
    triplet = []
    # head index should add1, as the root token would be the first token.
    # [ )  left bdr, right bdr.
    # left boundary, right boundary, parent, head

    for i, (parent, left_bdr, right_bdr) in enumerate(zip(heads, left_bd, right_bd)):
        triplet.append([left_bdr, right_bdr, parent-1, i])

    return triplet

import copy
def get_depth_first_pointing_action(headed_spans, label):
    from functools import cmp_to_key

    # pre-order sorting. e.g. [0, 4] [0, 2] [2, 3] [4, 5] [0, 1] [1, 2] -> [0, 4] [0, 2] [0, 1] [1, 2] [2, 3] [4, 5]
    def compare(a, b):
        if a[0] > b[0]:
            return 1
        elif a[0] == b[0]:
            if a[1] > b[1]:
                return -1
            else:
                return 1
        else:
            return -1

    headed_spans.sort(key=cmp_to_key(compare))

    # -1 stands for do not exist.
    span_end_boundary_pointer = []
    span_start_boundary_indicator = []
    child_word_pointer = []
    parent_triplet = []
    sibling_triplet = []
    grandparent_triplet = []
    stack = []
    new_label = []

    max_len = headed_spans[0][1]
    # begining 最开始的还是不要放进来了吧.
    parent_triplet.append([0, 0, max_len])
    sibling_triplet.append([0, 0, max_len+1])
    grandparent_triplet.append([0, 0, max_len+2])
    span_start_boundary_indicator.append(0)
    span_end_boundary_pointer.append(headed_spans[0][1])
    child_word_pointer.append(headed_spans[0][-1])
    new_label.append(label[headed_spans[0][-1]])

    stack.append(headed_spans.pop(0))
    last_generate = None

    # rule out sentence length1.
    if len(headed_spans) > 0:

        while len(stack) > 0:
            triplet = headed_spans.pop(0)
            assert triplet[0] >= stack[-1][0]
            assert triplet[1] <= stack[-1][1]
            span_start_boundary_indicator.append(triplet[0])
            span_end_boundary_pointer.append(triplet[1])
            child_word_pointer.append(triplet[-1])
            new_label.append(label[triplet[-1]])

            # parent一定是stack最后面一个element.
            parent_triplet.append([stack[-1][0], stack[-1][1], stack[-1][-1]])
            # 倒数第二个stack是grandparent, 如果没有则用-1来代替.
            if len(stack)>1:
                grandparent_triplet.append([stack[-2][0], stack[-2][1], stack[-2][-1]])
            else:
                grandparent_triplet.append([0, 0, max_len+2])
            #sibling怎么处理呢？最近一个被pop出stack的元素！那我们定义sibling是从左到右的顺序哦.
            # 如果连续push进去了两个，我们应该清空last_generate? check this.
            if last_generate is None:
                sibling_triplet.append([0, 0, max_len+3])
            else:
                sibling_triplet.append([last_generate[0], last_generate[1], last_generate[-1]])
            # 说明需要继续向下面迭代！
            if triplet[1] > triplet[0] + 1:
                stack.append(triplet)
                last_generate = None
            # 最小的span了. 一定要上去.
            else:
                last_generate = triplet
                if triplet[1] == stack[-1][1]:
                    while (len(stack) > 0) and (triplet[1] == stack[-1][1]) :
                        last_generate = stack.pop(-1)
                # in this case, the stack[-1]'s headword lying in the rightmost
                if (len(stack) > 0) and (triplet[1] == stack[-1][1] -1) and (stack[-1][-1] == stack[-1][1] - 1):
                    tmp = triplet[1] + 1
                    while (len(stack) > 0) and tmp == stack[-1][1]:
                        last_generate = stack.pop(-1)
                        if (len(stack) > 0) and (tmp == stack[-1][1] - 1) and (stack[-1][-1] == stack[-1][1] - 1):
                            tmp = stack[-1][1]
    assert len(headed_spans) == 0
    return span_start_boundary_indicator, span_end_boundary_pointer, child_word_pointer, parent_triplet, sibling_triplet, grandparent_triplet, new_label


def isProjective(heads):
    pairs = [(h, d) for d, h in enumerate(heads, 1) if h >= 0]
    for i, (hi, di) in enumerate(pairs):
        for hj, dj in pairs[i+1:]:
            (li, ri), (lj, rj) = sorted([hi, di]), sorted([hj, dj])
            if li <= hj <= ri and hi == dj:
                return False
            if lj <= hi <= rj and hj == di:
                return False
            if (li < lj < ri or li < rj < ri) and (li - lj)*(ri - rj) > 0:
                return False
    return True






