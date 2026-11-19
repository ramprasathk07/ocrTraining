"""Converters: each turns one dataset into CropRecords in its own output folder.

A converter is `convert(source: dict, out_dir: Path, workers: int) -> int` (records written).
`source` is that dataset's `[[sources]]` block from configs/data/training_data.toml.
Add a dataset = reuse a converter through config, or write one function and register it here.
"""

from dataEngine.sources import golden_regions, labelled_folder

CONVERTERS = {
    "golden_regions": golden_regions.convert,
    "labelled_folder": labelled_folder.convert,
}
