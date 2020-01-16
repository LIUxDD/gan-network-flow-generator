import os

from typing import Iterator, List
from struct import unpack as struct_unpack

import pandas as pd
import numpy as np

from network_flow_generator.utils.file_utils import ensure_file
from network_flow_generator.log import Logger

log = Logger.get()


class Preprocessor:

    def __init__(self, df: Iterator[pd.DataFrame]):
        self._df = df

    def save(self, fpath: str, force=False) -> None:
        """Saves the processed dataframe to a csv file.

        Args:
            fpath (str): The file path
            force (bool, optional): Overwrite an existing file. Defaults to False.

        Raises:
            OSError: If the given path exists and is not a file.
            FileExistsError: If file already exists and force is set to false.
        """
        ensure_file(fpath, force)

        first_chunk = True
        for df in self._df:
            df = self._process(df)
            df.to_csv(fpath, mode="a", compression="infer", header=first_chunk, index=False)
            first_chunk = False

    def get(self) -> Iterator[pd.DataFrame]:
        """Gets the processed dataframe.

        Returns:
            Iterator[pd.DataFrame]: Generator of processsed dataframes
        """
        for df in self._df:
            df = self._process(df)
            yield df

    def _process(self, df: pd.DataFrame):
        """Processes a single pandas dataframe.

        Args:
            df (pd.DataFrame): The dataframe to process
        """
        raise NotImplementedError()


class CiddsBinaryPreprocessor(Preprocessor):

    def _unpack_bits(self, number: int, bits: int) -> List[np.uint8]:
        """Converts a decimal number to a binary number.

        Args:
            number (int): A decimal number to convert
            bits (int): Number of bits the resulting binary number shall have.

        Returns:
            List[np.uint8]: Binary representation of the decimal number as a list of bits.
        """
        dtype = {
            8: ">i1",
            16: ">i2",
            32: ">i4",
        }.get(bits, ">i4")
        return np.unpackbits(np.array([number], dtype=dtype).view(np.uint8))

    def _process(self, df: pd.DataFrame):
        """Processes a single pandas dataframe in cidds format to be used with tensorflow.

        Usefull information: https://www.tensorflow.org/tutorials/structured_data/feature_columns

        Args:
            df (pd.DataFrame): The dataframe to process
        """
        log.debug("Processing dataframe")

        # date_first_seen
        indicies = [
            "isMonday",
            "isTuesday",
            "isWednesday",
            "isThursday",
            "isFriday",
            "isSaturday",
            "isSunday",
        ]
        dayofweek = df["date_first_seen"].dt.dayofweek
        weekdays = []
        for i, weekday in enumerate(indicies):
            weekdays.append((dayofweek == i).map(int).rename(weekday))
        daytime = (df["date_first_seen"] -
                   df["date_first_seen"].astype("datetime64[D]")).apply(lambda v: v.seconds / 86400)

        # duration
        # normalize over chunk
        min_duration = df["duration"].min()
        max_duration = df["duration"].max()
        norm_duration = ((df["duration"] - min_duration) / (max_duration - min_duration)).rename("norm_dur")

        # proto
        proto_tcp = (df["proto"] == "TCP").apply(int).rename("isTCP")
        proto_udp = (df["proto"] == "UDP").apply(int).rename("isUDP")
        proto_icmp = (df["proto"] == "ICMP").apply(int).rename("isICMP")

        # src_ip_addr
        indicies = ["src_ip_" + str(i) for i in range(32)]
        src_ip_addr = df["src_ip_addr"].apply(lambda v: pd.Series(self._unpack_bits(int(v), 32), index=indicies))

        # src_pt
        indicies = ["src_pt_" + str(i) for i in range(16)]
        src_pt = df["src_pt"].apply(lambda v: pd.Series(self._unpack_bits(int(v), 16), index=indicies))

        # dst_ip_addr
        indicies = ["dst_ip_" + str(i) for i in range(32)]
        dst_ip_addr = df["dst_ip_addr"].apply(lambda v: pd.Series(self._unpack_bits(int(v), 32), index=indicies))

        # dst_pt
        indicies = ["dst_pt_" + str(i) for i in range(16)]
        dst_pt = df["dst_pt"].apply(lambda v: pd.Series(self._unpack_bits(int(v), 16), index=indicies))

        # packets
        indicies = ["pck_" + str(i) for i in range(32)]
        packets = df["packets"].apply(lambda v: pd.Series(self._unpack_bits(int(v), 32), index=indicies))

        # bytes
        indicies = ["byt_" + str(i) for i in range(32)]
        _bytes = df["bytes"].apply(lambda v: pd.Series(self._unpack_bits(int(v), 32), index=indicies))

        # tcp flags
        indicies = ["isURG", "isACK", "isPSH", "isRES", "isSYN", "isFIN"]
        flags = df["flags"].apply(lambda v: pd.Series(map(int, v), index=indicies))

        # create DataFrame
        all_series = weekdays + [
            daytime, norm_duration, proto_tcp, proto_udp, proto_icmp, src_ip_addr, src_pt, dst_ip_addr, dst_pt, packets,
            _bytes, flags
        ]
        processed_df = pd.concat(all_series, axis=1)

        return processed_df


class CiddsNumericPreprocessor(Preprocessor):

    def _process(self, df: pd.DataFrame):
        """Processes a single pandas dataframe in cidds format to be used with tensorflow.

        Usefull information: https://www.tensorflow.org/tutorials/structured_data/feature_columns

        Args:
            df (pd.DataFrame): The dataframe to process
        """
        log.debug("Processing dataframe")

        indicies = [
            "isMonday",
            "isTuesday",
            "isWednesday",
            "isThursday",
            "isFriday",
            "isSaturday",
            "isSunday",
        ]
        dayofweek = df["date_first_seen"].dt.dayofweek
        weekdays = []
        for i, weekday in enumerate(indicies):
            weekdays.append((dayofweek == i).map(int).rename(weekday))
        daytime = (df["date_first_seen"] -
                   df["date_first_seen"].astype("datetime64[D]")).apply(lambda v: v.seconds / 86400)

        # duration
        # normalize over chunk
        min_duration = df["duration"].min()
        max_duration = df["duration"].max()
        norm_duration = ((df["duration"] - min_duration) / (max_duration - min_duration)).rename("norm_dur")

        # proto
        proto_tcp = (df["proto"] == "TCP").apply(int).rename("isTCP")
        proto_udp = (df["proto"] == "UDP").apply(int).rename("isUDP")
        proto_icmp = (df["proto"] == "ICMP").apply(int).rename("isICMP")

        # src_ip_addr
        indicies = ["src_ip_" + str(i) for i in range(4)]
        src_ip_addr = df["src_ip_addr"].apply(
            lambda v: pd.Series([x / 255 for x in struct_unpack('BBBB', v.packed)], index=indicies))

        # src_pt
        src_pt = df["src_pt"].apply(lambda v: v / 65535).rename("src_pt")

        # dst_ip_addr
        indicies = ["dst_ip_" + str(i) for i in range(4)]
        dst_ip_addr = df["dst_ip_addr"].apply(
            lambda v: pd.Series([x / 255 for x in struct_unpack('BBBB', v.packed)], index=indicies))

        # dst_pt
        dst_pt = df["dst_pt"].apply(lambda v: v / 65535).rename("dst_pt")

        # packets
        min_packets = df["packets"].min()
        max_packets = df["packets"].max()
        norm_packets = ((df["packets"] - min_packets) / (max_packets - min_packets)).rename("norm_pck")

        # bytes
        min_bytes = df["bytes"].min()
        max_bytes = df["bytes"].max()
        norm_bytes = ((df["bytes"] - min_bytes) / (max_bytes - min_bytes)).rename("norm_byt")

        # tcp flags
        indicies = ["isURG", "isACK", "isPSH", "isRES", "isSYN", "isFIN"]
        flags = df["flags"].apply(lambda v: pd.Series(map(int, v), index=indicies))

        # create DataFrame
        all_series = weekdays + [
            daytime, norm_duration, proto_tcp, proto_udp, proto_icmp, src_ip_addr, src_pt, dst_ip_addr, dst_pt,
            norm_packets, norm_bytes, flags
        ]
        processed_df = pd.concat(all_series, axis=1)

        return processed_df


class CiddsEmbeddingPreprocessor(Preprocessor):

    def _process(self, df: pd.DataFrame):
        """Processes a single pandas dataframe in cidds format to be used with tensorflow.

        Usefull information: https://www.tensorflow.org/tutorials/structured_data/feature_columns

        Args:
            df (pd.DataFrame): The dataframe to process
        """
        log.debug("Processing dataframe")

        indicies = [
            "isMonday",
            "isTuesday",
            "isWednesday",
            "isThursday",
            "isFriday",
            "isSaturday",
            "isSunday",
        ]
        dayofweek = df["date_first_seen"].dt.dayofweek
        weekdays = []
        for i, weekday in enumerate(indicies):
            weekdays.append((dayofweek == i).map(int).rename(weekday))
        daytime = (df["date_first_seen"] -
                   df["date_first_seen"].astype("datetime64[D]")).apply(lambda v: v.seconds / 86400)

        # duration
        duration = df["duration"]

        # proto
        proto_tcp = (df["proto"] == "TCP").apply(int).rename("isTCP")
        proto_udp = (df["proto"] == "UDP").apply(int).rename("isUDP")
        proto_icmp = (df["proto"] == "ICMP").apply(int).rename("isICMP")

        # src_ip_addr
        src_ip_addr = df["src_ip_addr"].astype("str")

        # src_pt
        src_pt = df["src_pt"]

        # dst_ip_addr
        dst_ip_addr = df["dst_ip_addr"].astype("str")

        # dst_pt
        dst_pt = df["dst_pt"]

        # packets
        packets = df["packets"]

        # bytes
        _bytes = df["bytes"]

        # tcp flags
        indicies = ["isURG", "isACK", "isPSH", "isRES", "isSYN", "isFIN"]
        flags = df["flags"].apply(lambda v: pd.Series(map(int, v), index=indicies))

        # create DataFrame
        all_series = weekdays + [
            daytime, duration, proto_tcp, proto_udp, proto_icmp, src_ip_addr, src_pt, dst_ip_addr, dst_pt, packets,
            _bytes, flags
        ]
        processed_df = pd.concat(all_series, axis=1)

        return processed_df
