import pickle

from bucket_wrapper import get_buckets
from object_wrapper import get_object


def load_pickled_dataframe(obj_key):
    bucket =  [b for b in get_buckets() if b.name == "math-with-words"][0]
    pkl_file = get_object(bucket, obj_key)
    df = pickle.loads(pkl_file)
    df.drop(columns=["Unnamed: 0"], inplace=True)
    return df


def load_pickled_object(obj_key):
    bucket =  [b for b in get_buckets() if b.name == "math-with-words"][0]
    pkl_file = get_object(bucket, obj_key)
    obj = pickle.loads(pkl_file)
    return obj
    

if __name__ == "__main__":
    # print(load_pickled_dataframe())
    pass