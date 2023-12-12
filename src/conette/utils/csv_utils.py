#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import os.path as osp

from pathlib import Path
from typing import Any, Iterable, Mapping, Union

from conette.utils.collections import dict_list_to_list_dict, list_dict_to_dict_list


def load_csv_dict(
    fpath: Union[str, Path],
    has_fieldnames: bool = True,
    cast: bool = False,
) -> dict[str, list[Any]]:
    data = load_csv_list(fpath, has_fieldnames, cast)
    data = list_dict_to_dict_list(data, None, True)
    return data


def load_csv_list(
    fpath: Union[str, Path],
    has_fieldnames: bool = True,
    cast: bool = False,
) -> list[dict[str, Any]]:
    with open(fpath, "r") as file:
        if has_fieldnames:
            reader = csv.DictReader(file)
            data = list(reader)
            if len(data) == 0:
                return []
        else:
            reader = csv.reader(file)
            data = list(reader)
            if len(data) == 0:
                return []
            default_fieldnames = list(map(str, range(len(data[0]))))
            data = [dict(zip(default_fieldnames, data_i)) for data_i in data]

    if not cast:
        return data

    outs = []
    for data_i in data:
        outs_i = {}
        for k, vs in data_i.items():
            try:
                vs_new = []
                for v in vs:
                    v = eval(v)
                    vs_new.append(v)
                outs_i[k] = vs_new
            except (SyntaxError, NameError):
                outs_i[k] = vs
        outs.append(outs_i)
    return outs


def save_csv_dict(
    data: Mapping[str, Iterable[Any]],
    fpath: Union[str, Path],
    overwrite: bool = True,
) -> None:
    data = dict(zip(data.keys(), map(list, data.values())))
    data_list = dict_list_to_list_dict(data)  # type: ignore
    save_csv_list(data_list, fpath, overwrite)


def save_csv_list(
    data: Iterable[Mapping[str, Any]],
    fpath: Union[str, Path],
    overwrite: bool = True,
) -> None:
    data = list(data)
    if len(data) <= 0:
        raise ValueError(f"Invalid argument {data=}. (found empty iterable)")
    if not overwrite and osp.isfile(fpath):
        raise FileExistsError("File already exists and argument overwrite is False.")

    with open(fpath, "w") as file:
        fieldnames = list(data[0].keys())
        writer = csv.DictWriter(file, fieldnames)
        writer.writeheader()
        writer.writerows(data)
