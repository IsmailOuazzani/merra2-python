# merra2-python

### Install
Install the dependencies:
```
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Download datasets
Follow GES DISC's [Data Access Instructions](https://disc.gsfc.nasa.gov/information/documents?title=Data%20Access).

Generate an EarthData Token at `https://urs.earthdata.nasa.gov/users/<your-username>/user_tokens`. And export it ans an env variable:
```
export EARTHDATA_TOKEN=<YOUR TOKEN>
```

To avoid repeating the export step in every new terminal, consider writing it to your `~/.bashrc`.


Find the dataset you are interested at: [https://disc.gsfc.nasa.gov/datasets](https://disc.gsfc.nasa.gov/datasets).
Click "Subset/Get Data", select "Get Original Files" (default option), then "Get Data". Download the list to the `dataset_lists` directory in this repo.


### Example usage
Solar dataset:
```
python data.py \
    dataset_lists/subset_M2T1NXRAD_5.12.4_20250421_173157_.txt \
    outputs \
    --variables SWGDN \
    --coords "-15.675690,27.322971 -15.664875,27.906086 -16.328276,27.319568 -16.327703,27.867828"
```

Precipitation dataset:
```
python data.py \
    dataset_lists/subset_GPM_3IMERGHHL_07_20250421_175336_.txt \
    outputs \
    --log-level DEBUG \
    --variables precipitation \
    --coords "-15.675690,27.322971 -15.664875,27.906086 -16.328276,27.319568 -16.327703,27.867828"
```

