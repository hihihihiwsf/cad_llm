from transforms import repr_entities

string_entities = '-11,29,-6,-23;-6,-23,5,-31,11,-18;'
entities = [
        ((-11, 29), (-6, -23)),
        ((-6, -23), (5, -31), (11, -18)),
    ]


def test_repr_entities():
    assert string_entities == repr_entities(entities), repr_entities(entities)
    print("success - test_repr_entities")


if __name__ == '__main__':
    test_repr_entities()
