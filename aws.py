import pickle

from bucket_wrapper import get_buckets
from object_wrapper import get_object


def load_pickled_dataframe():
    bucket =  [b for b in get_buckets() if b.name == "math-with-words"][0]
    obj_key = "attitudes_survey_translation_9_25.pkl"
    pkl_file = get_object(bucket, obj_key)
    df = pickle.loads(pkl_file)
    df.drop(columns=["Unnamed: 0"], inplace=True)
    return df


if __name__ == "__main__":
    print(load_pickled_dataframe())