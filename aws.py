import pickle

import streamlit as st
from botocore.exceptions import EndpointConnectionError


from bucket_wrapper import get_buckets
from object_wrapper import get_object


def get_s3_bucket():
    bucket = [b for b in get_buckets() if b.name == "math-with-words"][0]
    return bucket


@st.cache(hash_funcs={s3.Bucket: hash})
def load_pickled_dataframe(bucket, obj_key):
    try:
        # bucket = [b for b in get_buckets() if b.name == "math-with-words"][0]
        pkl_file = get_object(bucket, obj_key)
        df = pickle.loads(pkl_file)

    except EndpointConnectionError:
        # No internet! Load local file.
        df = pickle.load(open(obj_key, "rb"))
    
    df.drop(columns=["Unnamed: 0"], inplace=True)
    return df


def load_pickled_object(obj_key):
    try:
        bucket =  [b for b in get_buckets() if b.name == "math-with-words"][0]
        pkl_file = get_object(bucket, obj_key)
        obj = pickle.loads(pkl_file)

    except EndpointConnectionError:
        # No internet! Load local file.
        obj = pickle.loads(open(obj_key, "rb"))
 
    return obj
    

if __name__ == "__main__":
    # print(load_pickled_dataframe())
    pass