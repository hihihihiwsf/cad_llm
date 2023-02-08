from main import load_model, load_dataset


def test_load_model():
    tokenizer, model = load_model("byt5-base")
    assert model.num_parameters() == 581653248
    print("success - test_load_model")


def test_load_data():
    val_dataset = load_dataset("/Users/katzm/data/sg_normalized/sg_obj_val.npy")
    assert len(val_dataset) == 39344, f"Expected 39344 found {len(val_dataset)}"
    [val_dataset[i] for i in range(10)]  # process elements without breaking
    print("success - test_load_data")


if __name__ == '__main__':
    test_load_model()
    test_load_data()
