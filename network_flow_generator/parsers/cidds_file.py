import fileinput
import ipaddress

import numpy as np
import pandas as pd

from network_flow_generator.log import Logger
from network_flow_generator.utils import pandas_apply_parallel

log = Logger.get(__name__)


class FlowData:

    def __init__(self):
        self._data = None
        self._date_first_seen = None
        self._duration = None
        self._proto = None
        self._src_ip_addr = None
        self._src_pt = None
        self._dst_ip_addr = None
        self._dst_pt = None
        self._packets = None
        self._bytes = None
        self._flows = None
        self._flags = None
        self._tos = None
        self._class = None
        self._attack_type = None
        self._attack_id = None
        self._attack_description = None


class CiddsFile:

    # see https://docs.scipy.org/doc/numpy/reference/arrays.dtypes.html
    _data_types = [
        ("date_first_seen", "datetime64[ms]"),
        ("duration", "f4"),
        ("proto", "i1"),
        ("src_ip_addr", "object"),
        ("src_pt", "u2"),
        ("dst_ip_addr", "object"),
        ("dst_pt", "u2"),
        ("packets", "u8"),
        ("bytes", "u8"),
        ("flows", "u1"),
        ("flags", "?"),
        ("tos", "u1"),
        ("class", "i1"),
        ("attack_type", "U8"),
        ("attack_id", "U8"),
        ("attack_description", "U8"),
    ]

    _protocol_mapping = {
        "UDP": np.uint(1),
        "TCP": np.uint(2),
        "ICMP": np.uint(3),
    }

    _flow_class_mapping = {
        "normal": np.uint(1),
        "attacker": np.uint(2),
        "victim": np.uint(3),
        "suspicious": np.uint(4),
        "unknown": np.uint(5),
    }

    # random IPv4 addresses as replacements
    _ipv4_replacements = {
        "OPENSTACK_NET": "174.138.74.74",
        "DNS": "9.9.9.9",
        "EXT_SERVER": "220.175.38.139",
        "ATTACKER1": "230.170.204.100",
        "ATTACKER2": "185.135.146.33",
        "ATTACKER3": "201.95.169.48",
    }

    def __init__(self, path):
        self._path = path
        self._file = None

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()

    @classmethod
    def _convert_ipv4_address(cls, value):
        """Creates missing public ip addresses and converts them into ``IPv4Address`` datatypes.

        Args:
            value (str): The input value to convert.

        Returns:
            IPv4Address: Converted ip4v address.
        """
        # value = value.decode("utf-8")
        # anonymized ip addresses with special treatment
        if value in cls._ipv4_replacements:
            return ipaddress.IPv4Address(cls._ipv4_replacements[value])

        # other anonymized addresses
        if "_" in value:
            splitted_value = value.split("_")
            while True:
                # turn the random value into a pseudo-random ip address
                ipv4_address = ipaddress.IPv4Address((hash(splitted_value[0]) % 16777216) << 8) + int(splitted_value[1])
                if ipv4_address.is_global:
                    return ipv4_address
                else:
                    splitted_value[0] += "0"

        # regular ip addresses
        return ipaddress.IPv4Address(value)

    @classmethod
    def _convert_flags(cls, value):
        """Converts a string of TCP flags into an tuple of booleans.

        Args:
            value (str): Input value to convert.

        Returns:
            tuple of bool: Tuple of active flags: URG, ACK, PSH, RST, SYN, FIN
        """
        return (
            value[0:1] == "U",
            value[1:2] == "A",
            value[2:3] == "P",
            value[3:4] == "R",
            value[4:5] == "S",
            value[5:6] == "F",
        )

    @classmethod
    def _convert_flags_reverse(cls, value):
        """Converts a string of TCP flags into an tuple of booleans.

        Args:
            value (str): Input value to convert.

        Returns:
            tuple of bool: Tuple of active flags: URG, ACK, PSH, RST, SYN, FIN
        """

        return "".join([
            "U" if value[0] else ".",
            "A" if value[1] else ".",
            "P" if value[2] else ".",
            "R" if value[3] else ".",
            "S" if value[4] else ".",
            "F" if value[5] else ".",
        ])

    @classmethod
    def _convert_bytes(cls, value):
        try:
            return np.int64(value)
        except ValueError:
            # value has a format like "1.0 M"
            return np.int64(np.float(value.split()[0]) * 1e6)

    @classmethod
    def _convert_destination_port(cls, value):
        """Convert destination port to integer number. The destination port values of ICMP requests
            are expressed as float values so I need to take care of them somehow.

        Args:
            value (str): The input value

        Returns:
            np.int64: The parsed value
        """
        try:
            return np.int64(value)
        except ValueError:
            return np.int64(np.float(value) * 10)

    def _apply_converters(self, df):
        # apply converters
        df["src_ip_addr"] = df["src_ip_addr"].map(self._convert_ipv4_address)
        df["dst_ip_addr"] = df["dst_ip_addr"].map(self._convert_ipv4_address)
        df["dst_pt"] = df["dst_pt"].map(self._convert_destination_port).astype("uint16")
        # bytes can be either an integer or number suffixed with "M"
        df["bytes"] = df["bytes"].map(self._convert_bytes)
        df["flags"] = df["flags"].map(self._convert_flags)
        return df

    def read_chunks(self, chunksize=500000, nrows=None):
        headers = [
            "date_first_seen",
            "duration",
            "proto",
            "src_ip_addr",
            "src_pt",
            "dst_ip_addr",
            "dst_pt",
            "packets",
            "bytes",
            "flows",
            "flags",
            "tos",
            "class",
            "attack_type",
            "attack_id",
            "attack_description",
        ]

        dtypes = {
            "duration": "float32",
            "proto": "category",
            "src_pt": "uint16",
            "src_ip_addr": "str",
            "dst_ip_addr": "str",
            "dst_pt": "float64",
            "packets": "uint64",
            "flows": "uint8",
            "tos": "uint8",
            "class": "category",
            "attack_type": "category",
            "attack_id": "str",
            "attack_description": "str",
        }

        if chunksize:
            chunk_number = 0
            chunks = pd.read_csv(
                self._path,
                header=None,
                skiprows=1,
                names=headers,
                dtype=dtypes,
                parse_dates=["date_first_seen"],
                delimiter=",",
                error_bad_lines=False,
                nrows=nrows,
                chunksize=chunksize,
                low_memory=True)

            try:
                while True:
                    log.debug("Read chunk %d through %d of '%s'", chunk_number, chunksize + chunksize, self._path)
                    chunk = next(chunks)
                    yield chunk
                    chunk_number += chunksize
                    if chunk.shape[0] < chunksize:
                        return
            except StopIteration:
                return

        else:
            log.debug("Read dataframe from of '%s'", self._path)
            df = pd.read_csv(
                self._path,
                header=None,
                skiprows=1,
                names=headers,
                dtype=dtypes,
                parse_dates=["date_first_seen"],
                delimiter=",",
                error_bad_lines=False,
                nrows=nrows,
                low_memory=True)

            # apply data converters in parallel if the dataframe is big enough
            if df.shape[0] >= 25000:
                df = pandas_apply_parallel(df, self._apply_converters)
            else:
                df = self._apply_converters(df)

            return df

    @staticmethod
    def _read_files(files):
        with fileinput.input(files) as lines:
            for line in lines:
                if not fileinput.isfirstline():
                    yield line

    def write_chunk(self, chunk):
        if self._file is None:
            self._file = open(self._path, "w")
            # write header
            headers = [
                "Date first seen",
                "Duration",
                "Proto",
                "Src IP Addr",
                "Src Pt",
                "Dst IP Addr",
                "Dst Pt",
                "Packets",
                "Bytes",
                "Flows",
                "Flags",
                "Tos",
                "class",
                "attackType",
                "attackID",
                "attackDescription",
            ]
            self._file.write(",".join(headers) + "\n")

        log.debug("Write chunk to '%s'", self._path)
        chunk["flags"] = chunk["flags"].map(self._convert_flags_reverse)
        chunk.to_csv(self._file, header=False, index=False)

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None
