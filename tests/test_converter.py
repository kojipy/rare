def test_decode(converter):
    result = converter.decode([0, 1, 2, 3])
    assert ["[PAD]", "[GO]"] == result
