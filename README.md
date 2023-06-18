# CropNet: An Open Large-Scale Dataset with Multiple Modalities for Climate Change-aware Crop Yield Predictions

![Motivation](images/dataset-motivation.png)



The CropNet dataset is an open, large-scale, and deep learning-ready dataset, specifically targeting climate change-aware crop yield predictions for the contiguous United States (U.S.) continent at the county level. It is composed of three modalities of data, i.e., Sentinel-2 Imagery, WRF-HRRR Computed Dataset, and USDA Crop Dataset, aligned in both the spatial and temporal domains, for over 2200 U.S. counties spanning 6 years (2017-2022). It is expected to facilitate researchers in developing deep learning models for timely and precisely predicting crop yields at the county level, by accounting for the effects of both short-term growing season weather variations and long-term climate change on crop yields.

- **The dataset is available at [Google Drive](https://drive.google.com/drive/folders/1Js98GAxf1LeAUTxP1JMZZIrKvyJStDgz)**
- **The `CropNet` package is availbale at [The Python Package Index (PyPI)](https://pypi.org/project/cropnet/)**



## Overview

0ur CropNet dataset is composed of three modalities of data, i.e., Sentinel-2 Imagery, WRF-HRRR Computed Dataset, and USDA Crop Dataset, spanning from 2017 to 2022 (i.e., 6 years) across 2291 U.S. counties, with its geographic distribution illustrated below. We also include the number of counties corresponding to each crop type in the USDA Crop Dataset (see the rightmost bar chart in the figure) since crop planting is highly geography-dependent.

![Geographic Distribution](images/dataset-geo-overview-violet-pastel.png)

### Sentinel-2 Imagery

The Sentinel-2 Imagery, obtained from the Sentinel-2 mission, provides high-resolution satellite images for monitoring crop growth on the ground. It contains 224x224 RGB satellite images, with a spatial resolution of 9x9 km, and a revisit frequency of 14 days. The figures depict satellite images from the Sentinel-2 Imagery under four different revisit dates.

![Sentinel-2 Imagery](images/dataset-Sentinel-2-Imagery.png)



### WRF-HRRR Computed Dataset

The WRF-HRRR Computed Dataset, sourced from the WRF-HRRR model, contains daily and monthly meteorological parameters, with the former and the latter designed for capturing direct effects of short-term growing season weather variations on crop growth, and for learning indirect impacts of long-term climate change on crop yields, respectively. It contains 9 meteorological parameters gridded at 9 km in a one-day (and one-month) interval. The figures show the temperature in Spring, Summer, Fall, and Winter, respectively.

![HRRR Temperature](images/dataset-HRRR-temperature.png)



### USDA Crop Dataset

The USDA Crop Dataset, collected from the USDA Quick Statistic website, offers valuable information, such as production, yield, etc., for crops grown at each available county. It offers crop information for four types of crops, i.e., corn, cotton, soybeans, and winter wheat,  at a county-level basis, with a temporal resolution of one year. The figure illustrates the 2022 Corn Yield across the United States.

![USDA Corn Yield](images/dataset-corn-yield.png)



Although our initial goal of crafting the CropNet dataset is for precise crop yield prediction, we believe its future applicability is broad and can benefit the deep learning, agriculture, and meteorology communities, for exploring more interesting, critical, and climate change-related applications, by using one or more modalities of data.



### The CropNet Package

The code in the `CropNet` package

1. combines all three modalities of data to create $(\mathbf{x}, \mathbf{y_{s}}, \mathbf{y_{l}}, \mathbf{z})$ tuples, with $\mathbf{x}, \mathbf{y_{s}}, \mathbf{y_{l}}, \text{and}~ \mathbf{z}$ representing satellite images, short-term daily whether parameters, long-term monthly meterological parameters, and ground-truth crop yield (or production) inforamtion, resprectively, and
2. exposes those tuples via a `Dataset` object.

Notably, one or more modalities of data can be used for specific deep learning tasks. For example,

1. satellite images can be solely utilized for pre-training deep neural networks in a self-supervised learning manner (e.g., [SimCLR](https://arxiv.org/pdf/2002.05709.pdf), [MAE](https://arxiv.org/pdf/2111.06377.pdf), etc.), or
2. a pair of $(\mathbf{x}, \mathbf{y_{s}})$ under the same 9x9 km grid can be used for exploring the local weather effect on crop growth.



### Installation

- MacOS and Linux users can install the latest version of CropNet with the following command:

```sh
pip install cropnet
```



## License

CropNet has a [Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/) license.