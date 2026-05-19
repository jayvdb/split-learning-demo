from functools import wraps
from typing import Any, Callable, Dict, List, Union

from datasets import (
    Dataset,
    DatasetDict,
    Image,
    IterableDataset,
    IterableDatasetDict,
    load_dataset,
)

from split_learning.utils import utils

# huggingface datasets


def dataset_loader(
    name: str,
    format_=None,
    format_columns_: List[str] = None,
    transform_: Callable[[Any], Any] = None,
    transform_columns_: List[str] = None,
    map_func_: Callable[[Any], Any] = None,
    map_kwargs_: Dict[str, Any] = None,
    rename_columns_: Dict[str, str] = None,
    **load_dataset_kwargs_,
):
    """Decorator for loading datasets from huggingface datasets."""

    def decorator(
        func: Callable[
            [Any], Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]
        ]
    ):
        @wraps(func)
        def wrapper(
            format=None,
            format_columns=None,
            transform: Callable[[Any], Any] = None,
            transform_columns: List[str] = None,
            map_func: Callable[[Any], Any] = None,
            map_kwargs: Dict[str, Any] = {},
            rename_columns: Dict[str, str] = None,
            **load_dataset_kwargs,
        ):
            format = format_ if format is None else format
            format_columns = (
                format_columns_ if format_columns is None else format_columns
            )
            transform = transform_ if transform is None else transform
            transform_columns = (
                transform_columns_ if transform_columns is None else transform_columns
            )
            map_func = map_func_ if map_func is None else map_func
            map_kwargs = map_kwargs_ if map_kwargs is None else map_kwargs
            rename_columns = (
                rename_columns_ if rename_columns is None else rename_columns
            )
            load_dataset_kwargs = {**load_dataset_kwargs_, **load_dataset_kwargs}

            # load dataset
            data_dir = utils.data_path() / "external" / name
            hf_load_dataset_kwargs = {"cache_dir": data_dir, **load_dataset_kwargs}
            dataset = load_dataset(name, **hf_load_dataset_kwargs)

            # rename columns
            if rename_columns is not None:
                dataset = dataset.rename_columns(rename_columns)

            # downstream preprocessing dataset
            if map_func is not None:

                def map_fn(example, *args, **kwargs):
                    if map_func is not None:
                        example = map_func(example, *args, **kwargs)
                    return example

                hf_map_kwargs = {"num_proc": 4, **map_kwargs}
                dataset = dataset.map(map_fn, **hf_map_kwargs)

            # set format
            hf_format_kwargs = {"columns": format_columns}
            if format is not None:
                dataset.set_format(format, **hf_format_kwargs)
            elif transform is not None:

                def hf_transforms(examples):
                    for column in transform_columns:
                        examples[column] = [
                            transform(image) for image in examples[column]
                        ]
                    return examples

                dataset.set_transform(hf_transforms, **hf_format_kwargs)

            return func(dataset)

        return wrapper

    return decorator


def mnist(
    split: str = None,
    transform: Callable[[Any], Any] = None,
    transform_columns: List[str] = None,
):
    """Load the MNIST dataset from the pre-fetched parquet files.

    Run ``python scripts/fetch_data.py --dataset mnist`` once to populate
    ``data/external/mnist/`` (parquet from the ``ylecun/mnist`` HF repo).
    """
    data_dir = utils.data_path() / "external" / "mnist"
    train_files = sorted(data_dir.rglob("train-*.parquet"))
    test_files = sorted(data_dir.rglob("test-*.parquet"))
    if not train_files or not test_files:
        raise FileNotFoundError(
            f"MNIST parquet not found under {data_dir}. "
            "Run: python scripts/fetch_data.py --dataset mnist"
        )

    train_paths = [str(p) for p in train_files]
    test_paths = [str(p) for p in test_files]

    # Use Dataset.from_parquet directly instead of load_dataset("parquet", ...).
    # The latter looks up a `parquet.py` dataset script on the legacy HF s3
    # bucket every call (a 404, ignored) before falling back to the in-library
    # builder; from_parquet skips that lookup and stays fully offline.
    if split == "train":
        dataset = Dataset.from_parquet(train_paths)
    elif split == "test":
        dataset = Dataset.from_parquet(test_paths)
    elif split is None:
        dataset = DatasetDict(
            {
                "train": Dataset.from_parquet(train_paths),
                "test": Dataset.from_parquet(test_paths),
            }
        )
    else:
        raise ValueError(f"Unknown split: {split!r} (expected 'train', 'test', or None)")

    # Parquet preserves the image column as raw bytes; cast it back so the
    # transform pipeline sees PIL.Image like the previous loader did.
    dataset = dataset.cast_column("image", Image())

    if transform is not None:
        cols = transform_columns or ["image"]

        def hf_transforms(examples):
            for column in cols:
                examples[column] = [transform(image) for image in examples[column]]
            return examples

        # Don't pass columns=cols here — set_transform's columns arg restricts
        # which columns are *loaded* before transform, so the label would be
        # filtered out of the output. The transform itself only mutates `cols`.
        dataset.set_transform(hf_transforms)

    return dataset


@dataset_loader("fashion_mnist", transform_columns_=["image"])
def fashion_mnist(dataset):
    return dataset


@dataset_loader(
    "cifar10", transform_columns_=["image"], rename_columns_={"img": "image"}
)
def cifar10(dataset):
    return dataset


@dataset_loader("quickdraw", transform_columns_=["image"])
def quickdraw(dataset):
    return dataset


@dataset_loader("lansinuote/gen.1.celeba", transform_columns_=["image"])
def celeba(dataset):
    return dataset


@dataset_loader("imagenet-1k", transform_columns_=["image"])
def imagenet_1k(dataset):
    return dataset


@dataset_loader("zh-plus/tiny-imagenet", transform_columns_=["image"])
def tiny_imagenet(dataset):
    return dataset
