# Ubiquant Market Prediction

*Make predictions against future market data*

## Overview



### Description

Regardless of your investment strategy, fluctuations are expected in the financial market. Despite this variance, professional investors try to estimate their overall returns. Risks and returns differ based on investment types and other factors, which impact stability and volatility. To attempt to predict returns, there are many computer-based algorithms and models for financial market trading. Yet, with new techniques and approaches, data science could improve quantitative researchers' ability to forecast an investment's return.

![img](https://storage.googleapis.com/kaggle-media/competitions/ubiquant/6.jpg)

Ubiquant Investment (Beijing) Co., Ltd is a leading domestic quantitative hedge fund based in China. Established in 2012, they rely on international talents in math and computer science along with cutting-edge technology to drive quantitative financial market investment. Overall, Ubiquant is committed to creating long-term stable returns for investors.

In this competition, you’ll build a model that forecasts an investment's return rate. Train and test your algorithm on historical prices. Top entries will solve this real-world data science problem with as much accuracy as possible.

If successful, you could improve the ability of quantitative researchers to forecast returns. This will enable investors at any scale to make better decisions. You may even discover you have a knack for financial datasets, opening up a world of new opportunities in many industries.



### Evaluation

Submissions are evaluated on the mean of the [Pearson correlation coefficient](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient) for each time ID. 

You must submit to this competition using the provided python time-series API, which ensures that models do not peek forward in time. To use the API, follow this template in Kaggle Notebooks:

```
import ubiquant
env = ubiquant.make_env()   # initialize the environment
iter_test = env.iter_test()    # an iterator which loops over the test set and sample submission
for (test_df, sample_prediction_df) in iter_test:
    sample_prediction_df['target'] = 0  # make your predictions here
    env.predict(sample_prediction_df)   # register your predictions
```

You will get an error if you submission includes nulls or infinities and submissions that only include one prediction value will receive a score of -1.



## Data

Kaggle: https://www.kaggle.com/c/ubiquant-market-prediction/data

### Data Description

This dataset contains features derived from real historic data from thousands of investments. Your challenge is to predict the value of an obfuscated metric relevant for making trading decisions.

This is a code competition that relies on a time-series API to ensure models do not peek forward in time. To use the API, follow the instructions on the Evaluation page. When you submit your notebook, it will be rerun on an unseen test. This is also a forecasting competition, where the final private leaderboard will be determined using data gathered after the training period closes, which means that the public and private leaderboards will have zero overlap.

### Files

**train.csv**

- `row_id` - A unique identifier for the row.
- `time_id` - The ID code for the time the data was gathered. The time IDs are in order, but the real time between the time IDs is not constant and will likely be shorter for the final private test set than in the training set.
- `investment_id` - The ID code for an investment. Not all investment have data in all time IDs.
- `target` - The target.
- `[f_0:f_299]` - Anonymized features generated from market data.

**example_test.csv** - Random data provided to demonstrate what shape and format of data the API will deliver to your notebook when you submit.

**example_sample_submission.csv** - An example submission file provided so the publicly accessible copy of the API provides the correct data shape and format.

**ubiquant/** - The image delivery API that will serve the test set. You may need Python 3.7 and a Linux environment to run the example test set through the API offline without errors.

Time-series API Details - The API serves the data in batches, with all of rows for a single time `time_id` per batch.

- Expect to see roughly one million rows in the test set.
- The API will require roughly 0.25 GB of memory after initialization. The initialization step (env.iter_test()) will require meaningfully more memory than that; we recommend you do not load your model until after making that call.
- The API will also use less than 15 minutes of runtime for loading and serving the data.

```python
kaggle competitions download -c ubiquant-market-prediction
```

