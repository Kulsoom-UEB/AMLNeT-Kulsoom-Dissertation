# data/raw/

**This folder is intentionally empty in the repository.**

The AMLNet dataset is approximately 691 MB, which exceeds GitHub's 100 MB per-file
limit, so it is not committed here. It must be downloaded from its original source.

## How to obtain the dataset

1. Go to **https://doi.org/10.5281/zenodo.16736515**
2. Download `AMLNet.csv`
3. Place it in this folder, so the path is `data/raw/AMLNet.csv`

## Verifying the download

| Property | Expected value |
|---|---|
| Filename | `AMLNet.csv` |
| Size | 691.3 MB (659.3 MiB) |
| MD5 | `7668fc7d74c787e07546ce85c6f790b9` |
| Rows | 1,090,173 |
| Columns | 17 |

Running `python src/step00_setup.py` checks the file against this MD5 automatically
and stops with an error if it does not match, so a corrupted or partial download is
caught before any processing begins.

## Citation and licence

> Huda, S. (2025). *AMLNet: Synthetic anti-money laundering transaction dataset.*
> Zenodo. https://doi.org/10.5281/zenodo.16736515

Licensed **CC BY-NC 4.0** — non-commercial use with attribution. This project uses
the dataset strictly for academic research.
