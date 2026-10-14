from io import StringIO

from .vocab import Vocab


def test_dump_load():
    v = Vocab()
    v.add("甲", 3, "n")
    v.add("甲乙", 2)
    v.add("甲乙丙", 1)
    v.add("乙", 2)
    buf = StringIO()
    v.dumpfile(buf)
    buf.seek(0)
    v2 = Vocab.loadfile(buf)
    assert v == v2
