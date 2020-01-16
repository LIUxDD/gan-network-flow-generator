import types

import pandas as pd


def filter(data, query):
    if isinstance(data, pd.io.parsers.TextFileReader):
        for chunk in data:
            yield chunk[query]
    else:
        return data[query]


def _filter_chunk_by_day(chunk, day):
    # return chunk[chunk.date_first_seen - date < np.timedelta64(1, "D") and chunk.date_first_seen - date < np.timedelta64(1, "D")]
    helper = chunk.date_first_seen.astype("datetime64[D]")
    return chunk[helper == day]


def filter_by_day(data, day):
    day = day.astype("datetime64[D]")
    if isinstance(data, types.GeneratorType):
        for chunk in data:
            yield _filter_chunk_by_day(chunk, day)
    else:
        return _filter_chunk_by_day(data, day)
