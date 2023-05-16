"""

Script to convert DeepMind data from protocol buffers to json
Requires environment with tensorflow and protobuf

https://github.com/deepmind/deepmind-research/tree/master/cadl

"""

# import tensorflow as tf
# import example_pb2
# from google.protobuf import json_format
# import json
# from tqdm import tqdm
#
#
# def save_data(split, index, data):
#     path = f"data/{split}_{index:03}.json"
#     print(f"Saving to {path}")
#     with open(path, "w") as f:
#         json.dump(data, f)
#
#
# def convert_split(split):
#     dataset = tf.data.TFRecordDataset(f"data/{split}.tfrecord")
#
#     chunk_size = 50000
#     i = 0
#     chunk_index = 0
#     chunk_data = []
#
#     for raw_record in tqdm(dataset.as_numpy_iterator()):
#         example = example_pb2.Example()
#         example.ParseFromString(raw_record)
#         json_dict = json_format.MessageToDict(example)
#         chunk_data.append(json_dict)
#         i += 1
#         if i % chunk_size == 0:
#             save_data(split=split, index=chunk_index, data=chunk_data)
#             chunk_index += 1
#             chunk_data = []
#
#     save_data(split=split, index=chunk_index, data=chunk_data)
#
#
# def convert_data():
#     for split in ["train", "valid", "test"]:
#         convert_split(split)
#
#
# if __name__ == "__main__":
#     convert_data()
