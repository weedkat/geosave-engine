# Download benchmark files

```python
from geosave_engine.geodata.benchmarks import dynamic_world

labels = dynamic_world.download("data/dynamic_world", subset="experts")
# pathlib.Path('data/dynamic_world/experts')
```

The downloader extracts original publisher files and accompanying metadata.
It preserves folder layout and label values. It does not fetch source imagery,
convert metadata to a catalog, or make a training split.

| Subset | Release | Archive size, approximately |
| --- | --- | --- |
| `experts` | Expert training labels | 47.8 MB |
| `non_expert` | Commissioned training labels | 214.7 MB |
| `validation` | Individual expert labels and composites | 29 MB |
| `test` | Expert consensus, predicted labels, probabilities | 3.9 GB |

Training downloads also include the publisher README and main metadata
spreadsheet. Validation keeps its archive contents and adds the README. Original
Sentinel-2 imagery is absent from the training release; metadata identifies the
source scenes. The training data is CC BY 4.0; use the publisher's citation and
attribution. [PANGAEA training release](https://doi.pangaea.de/10.1594/PANGAEA.933475)

The test archive contains metadata CSV files. In its `label_*` rasters, `lulc`
is human annotation and `label` is a model prediction; probability rasters are
separate. Those bands must not be treated interchangeably as ground truth.
The download verifies the published archive MD5 checksum.
[Zenodo test release](https://zenodo.org/records/4766508)

## Reuse and interrupted downloads

```python
validation = dynamic_world.download("data/dynamic_world", subset="validation")
```

Each subset has its own directory and a `.geosave-complete.json` record.
Repeated calls reuse a matching completed extraction without network requests.
Transfers and extraction happen in a temporary directory; failures propagate and
do not publish a completed subset. Downloaded ZIP files are removed after
extraction. Budget disk space for both the archive and extracted files during
the download.

An existing destination without a matching completion record raises
`FileExistsError`. Move or remove that incomplete directory explicitly before
retrying. Malformed completion JSON raises `ValueError` without changing files.
A completion record tracks successful extraction; it does not repair
files edited or deleted afterward. The downloader is for local storage and does
not coordinate concurrent writers to the same subset directory.
