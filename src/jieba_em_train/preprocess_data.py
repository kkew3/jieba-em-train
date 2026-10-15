import jieba


def split_blocks(sentence: str):
    """Split `sentence` into jieba-cuttable blocks."""
    sentence = jieba.strdecode(sentence)
    for blk in jieba.re_han_default.split(sentence):
        if blk and jieba.re_han_default.match(blk):
            yield blk
