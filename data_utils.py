import numpy as np
import pandas as pd

from polara import get_movielens_data
from polara.preprocessing.dataframes import reindex, leave_one_out


def prepare_ml_data(path: str):
    ml_data, genres_info = get_movielens_data(
        local_file=path, include_time=True, get_genres=True
    )

    ml_data = ml_data.sort_values(by='timestamp').reset_index(drop=True)

    user_map = {_id : i for i, _id in enumerate(ml_data['userid'].unique())}
    item_map = {_id : i for i, _id in enumerate(genres_info['movieid'].unique())}

    ml_data['userid'] = ml_data['userid'].map(user_map)
    ml_data['movieid'] = ml_data['movieid'].map(item_map)
    genres_info['movieid'] = genres_info['movieid'].map(item_map)

    genres = genres_info['genreid'].unique()

    items_info = genres_info.groupby('movieid')['genreid']\
        .apply(lambda x: np.isin(genres, x).astype(int))
    item_vectors = np.stack(items_info.values)

    user_ids = ml_data['userid'].unique()
    item_ids = ml_data['movieid'].unique()

    user_sequences = ml_data.groupby('userid')['movieid'].apply(list)

    return user_ids, item_ids, item_vectors, user_sequences


def transform_indices(data, users, items):
    data_index = {}
    for entity, field in zip(['users', 'items'], [users, items]):
        idx, idx_map = to_numeric_id(data, field)
        data_index[entity] = idx_map
        data.loc[:, field] = idx
    return data, data_index

def to_numeric_id(data, field):
    idx_data = data[field].astype("category")
    idx = idx_data.cat.codes
    idx_map = idx_data.cat.categories.rename(field)
    return idx, idx_map


def get_dataset(
    validation_size=1024,
    test_size=5000,
    verbose=False,
    data_path=None,
    path=None,
    splitting="temporal_full",
    q=0.8,
):
    if type(path) == pd.core.frame.DataFrame:
        mldata = path
    elif isinstance(path, str) and path.endswith(".csv"):
        mldata = pd.read_csv(path)
    else:
        mldata = get_movielens_data(local_file=data_path, include_time=True).rename(
            columns={"movieid": "itemid"}
        )

    if splitting == "temporal_full":
        test_timepoint = mldata["timestamp"].quantile(q=q, interpolation="nearest")
        test_data_ = mldata.query("timestamp >= @test_timepoint")
        if verbose:
            print(test_data_.nunique())

        train_data_ = mldata.query(
            "timestamp < @test_timepoint"
        )  # убрал userid not in @test_data_.userid.unique() and
        training_, data_index = transform_indices(
            train_data_.copy(), "userid", "itemid"
        )

        testset_ = reindex(test_data_, data_index["items"])

        testset_valid_, holdout_valid_ = leave_one_out(
            training_, target="timestamp", sample_top=True, random_state=0
        )

        testset_valid, data_index = transform_indices(
            testset_valid_.copy(), "userid", "itemid"
        )

        testset_ = reindex(testset_, data_index["items"])
        holdout_valid = reindex(holdout_valid_, data_index["items"])
        holdout_valid = reindex(holdout_valid, data_index["users"]).sort_values(
            ["userid"]
        )
        if verbose:
            print(testset_.nunique())

        training = testset_valid.copy()

        validation_users = np.intersect1d(
            holdout_valid["userid"].unique(), testset_valid["userid"].unique()
        )
        if validation_size < len(validation_users):
            validation_users = np.random.choice(
                validation_users, size=validation_size, replace=False
            )
        testset_valid = testset_valid[
            testset_valid["userid"].isin(validation_users)
        ].sort_values(by=["userid", "timestamp"])
        holdout_valid = holdout_valid[holdout_valid["userid"].isin(validation_users)]

        testset_, holdout_ = leave_one_out(
            testset_, target="timestamp", sample_top=True, random_state=0
        )

        test_users = np.intersect1d(
            holdout_["userid"].unique(), testset_["userid"].unique()
        )
        if test_size < len(test_users):
            test_users = np.random.choice(test_users, size=test_size, replace=False)
        testset_ = testset_[testset_["userid"].isin(test_users)].sort_values(
            by=["userid", "timestamp"]
        )
        holdout_ = holdout_[holdout_["userid"].isin(test_users)].sort_values(["userid"])

    elif splitting == "temporal":
        test_timepoint = mldata["timestamp"].quantile(q=0.95, interpolation="nearest")
        test_data_ = mldata.query("timestamp >= @test_timepoint")
        if verbose:
            print(test_data_.nunique())

        train_data_ = mldata.query(
            "userid not in @test_data_.userid.unique() and timestamp < @test_timepoint"
        )
        training, data_index = transform_indices(train_data_.copy(), "userid", "itemid")

        test_data = reindex(test_data_, data_index["items"])
        if verbose:
            print(test_data.nunique())
        testset_, holdout_ = leave_one_out(
            test_data, target="timestamp", sample_top=True, random_state=0
        )
        testset_valid_, holdout_valid_ = leave_one_out(
            testset_, target="timestamp", sample_top=True, random_state=0
        )

        userid = data_index["users"].name
        test_users = pd.Index(
            # ensure test users are the same across testing data
            np.intersect1d(
                testset_valid_[userid].unique(), holdout_valid_[userid].unique()
            )
        )
        testset_valid = (
            testset_valid_
            # reindex warm-start users for convenience
            .assign(**{userid: lambda x: test_users.get_indexer(x[userid])})
            .query(f"{userid} >= 0")
            .sort_values("userid")
        )
        holdout_valid = (
            holdout_valid_
            # reindex warm-start users for convenience
            .assign(**{userid: lambda x: test_users.get_indexer(x[userid])})
            .query(f"{userid} >= 0")
            .sort_values("userid")
        )

        testset_ = (
            testset_
            # reindex warm-start users for convenience
            .assign(**{userid: lambda x: test_users.get_indexer(x[userid])})
            .query(f"{userid} >= 0")
            .sort_values("userid")
        )
        holdout_ = (
            holdout_
            # reindex warm-start users for convenience
            .assign(**{userid: lambda x: test_users.get_indexer(x[userid])})
            .query(f"{userid} >= 0")
            .sort_values("userid")
        )

    elif splitting == "leave-one-out":
        mldata, data_index = transform_indices(mldata.copy(), "userid", "itemid")
        training, holdout_ = leave_one_out(
            mldata, target="timestamp", sample_top=True, random_state=0
        )
        training_valid_, holdout_valid_ = leave_one_out(
            training, target="timestamp", sample_top=True, random_state=0
        )

        testset_valid_ = training_valid_.copy()
        testset_ = training.copy()
        training = training_valid_.copy()

        userid = data_index["users"].name
        test_users = pd.Index(
            # ensure test users are the same across testing data
            np.intersect1d(
                testset_valid_[userid].unique(), holdout_valid_[userid].unique()
            )
        )
        testset_valid = (
            testset_valid_
            # reindex warm-start users for convenience
            .assign(**{userid: lambda x: test_users.get_indexer(x[userid])})
            .query(f"{userid} >= 0")
            .sort_values("userid")
        )
        holdout_valid = (
            holdout_valid_
            # reindex warm-start users for convenience
            .assign(**{userid: lambda x: test_users.get_indexer(x[userid])})
            .query(f"{userid} >= 0")
            .sort_values("userid")
        )

        testset_ = (
            testset_
            # reindex warm-start users for convenience
            .assign(**{userid: lambda x: test_users.get_indexer(x[userid])})
            .query(f"{userid} >= 0")
            .sort_values("userid")
        )
        holdout_ = (
            holdout_
            # reindex warm-start users for convenience
            .assign(**{userid: lambda x: test_users.get_indexer(x[userid])})
            .query(f"{userid} >= 0")
            .sort_values("userid")
        )

    else:
        raise ValueError

    if verbose:
        print(testset_valid.nunique())
        print(holdout_valid.shape)
    # assert holdout_valid.set_index('userid')['timestamp'].ge(
    #     testset_valid
    #     .groupby('userid')
    #     ['timestamp'].max()
    # ).all()

    data_description = dict(
        users=data_index["users"].name,
        items=data_index["items"].name,
        order="timestamp",
        n_users=len(data_index["users"]),
        n_items=len(data_index["items"]),
    )

    if verbose:
        print(data_description)

    return training, data_description, testset_valid, testset_, holdout_valid, holdout_
