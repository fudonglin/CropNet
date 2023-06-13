# CropNet: An Open Large-Scale Dataset with Multiple Modalities for Climate Change-aware Crop Yield Predictions

Precise crop yield predictions are of national importance for ensuring food security and sustainable agricultural practices. While AI-for-science approaches have exhibited promising achievements in solving many scientific applications such as drug discovery, precipitation nowcasting, etc., the development of deep learning models for predicting crop yields is constantly hindered by the lack of an open and large-scale deep learning-ready dataset with multiple modalities having sufficient information. To remediate this, we aim to introduce the CropNet dataset, the first terabyte-sized, publicly available, and multi-modal dataset specifically targeting crop yield predictions for the contiguous United States (U.S.) continent at the county level. Our CropNet dataset is composed of three modalities of data, i.e., Sentinel-2 Imagery, WRF-HRRR Computed Dataset, and USDA Crop Dataset, for over 2200 U.S. counties spanning 6 years (2017-2022). Specifically, Sentinel-2 Imagery is obtained from the Sentinel-2 mission for precisely monitoring and relating crop growth. WRF-HRRR Computed Dataset, sourced from the Weather Research & Forecasting-based High-Resolution Rapid Refresh (WRF-HRRR) model, covers daily and monthly weather conditions, accounting respectively for growing season weather variations and climate change. USDA Crop Dataset, collected from the USDA Quick Statistic website, provides annual crop information for major crops grown in the U.S. Our CropNet is expected to facilitate researchers in developing versatile deep models for timely and precisely predicting crop yields at the county level, by accounting for the effects of both short-term growing season weather variations and long-term climate change on crop yields. We have conducted extensive experiments using our CropNet dataset, by employing a convolutional LSTM-based model and a Vision Transformer (ViT)-based model. The results validate the applicability of the CropNet dataset under different deep learning models and its efficacy in climate change-aware crop yield predictions.

![Motivation](images/dataset-motivation.png)



## Overview

0ur CropNet dataset is composed of three modalities of data, i.e., Sentinel-2 Imagery, WRF-HRRR Computed Dataset, and USDA Crop Dataset, spanning from 2017 to 2022 (i.e., 6 years) across 2291 U.S. counties, with its geographic distribution illustrated below. We also include the number of counties corresponding to each crop type in the USDA Crop Dataset (see the rightmost bar chart in the below figure) since crop planting is highly geography-dependent.

![Geographic Distribution](images/dataset-geo-overview-violet-pastel.png)



### Sentinel-2 Imagery

The Sentinel-2 Imagery, obtained from the Sentinel-2 mission, provides high-resolution satellite images for monitoring crop growth on the ground. It contains 1284.8 GB of 224x224 RGB satellite images, with a spatial resolution of 9x9 km, and a revisit frequency of 14 days. The figures depict satellite images from the Sentinel-2 Imagery under four different revisit dates.

![Sentinel-2 Imagery](images/dataset-Sentinel-2-Imagery.png)



### WRF-HRRR Computed Dataset

The WRF-HRRR Computed Dataset, sourced from the WRF-HRRR model, contains daily and monthly meteorological parameters, with the former and the latter designed for capturing the direct effects of short-term growing season weather variations on crop growth, and for learning the indirect impacts of long-term climate change on crop yields, respectively. It contains 9 meteorological parameters gridded at 9 km in a one-day (and one-month) interval, arriving at a total size of 35.5GB. The figures show the temperature in Spring, Summer, Fall, and Winter, respectively.

![HRRR Temperature](images/dataset-HRRR-temperature.png)



### USDA Crop Dataset

The USDA Crop Dataset, collected from the USDA Quick Statistic website, offers valuable information, such as production, yield, etc., for crops grown at each available county. It offers crop information for four types of crops at each county-level basis, with a temporal resolution of one year. The figure illustrates the 2022 Corn Yield across the United States.

![USDA Corn Yield](images/dataset-corn-yield.png)