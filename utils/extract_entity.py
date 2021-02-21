# coding=utf-8
# 严格按照BMOES一致类型抽取实体
def extract_kvpairs_in_bmoes(bmoes_seq, word_seq):
    assert len(bmoes_seq) == len(word_seq)
    pairs = list()
    pre_bmoes = "O"
    v = ""
    spos = -1
    for i, bmoes in enumerate(bmoes_seq):
        word = word_seq[i]
        if bmoes == "O":
            v = ""
        elif bmoes[0] == "B":
            v = word[2:] if word.startswith('##') else word
            spos = i
        elif bmoes[0] == "M":
            if pre_bmoes[0] in "OES" or pre_bmoes[2:] != bmoes[2:] or v == "":
                v = ""
            else:
                v += word[2:] if word.startswith('##') else word
        elif bmoes[0] == 'E':
            if pre_bmoes[0] in 'BM' and pre_bmoes[2:] == bmoes[2:] and v != "":
                v += word[2:] if word.startswith('##') else word
                pairs.append(((spos, i + 1), bmoes[2:], v))
            v = ""
        elif bmoes[0] == 'S':
            v = word[2:] if word.startswith('##') else word
            pairs.append(((i, i + 1), bmoes[2:], v))
            v = ""
        pre_bmoes = bmoes
    return pairs


# 严格按照BIO一致类型抽取实体
def extract_kvpairs_in_bio(bio_seq, word_seq):
    assert len(bio_seq) == len(word_seq)
    pairs = list()
    pre_bio = "O"
    v = ""
    spos = -1
    for i, bio in enumerate(bio_seq):
        word = word_seq[i]
        if bio == "O":
            if v != "":
                pairs.append(((spos, i), pre_bio[2:], v))
            v = ""
        elif bio[0] == "B":
            if v != "":
                pairs.append(((spos, i), pre_bio[2:], v))
            v = word[2:] if word.startswith('##') else word
            spos = i
        elif bio[0] == "I":
            if pre_bio[0] == "O" or pre_bio[2:] != bio[2:] or v == "":
                if v != "":
                    pairs.append(((spos, i), pre_bio[2:], v))
                v = ""
            else:
                v += word[2:] if word.startswith('##') else word
        pre_bio = bio
    if v != "":
        pairs.append(((spos, len(bio_seq)), pre_bio[2:], v))
    return pairs
