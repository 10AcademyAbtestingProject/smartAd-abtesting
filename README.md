# smartAd-abtesting

- SmartAd is an advertisement company.
- They Create Ads for client and charge based on user engagement.
- They also quantify the increase in brand awareness as a result of Ads shown. This is their Brand Impact Optimizer BMO.
- The company is based on the principle of voluntary participation.
- This has been proven to increase brand awareness and memorability 10X more than static options.

## BMO (Brand Impact Optimizer)

you should read more on [here](https://www.booyahadvertising.com/blog/remember-these-top-of-the-funnel-marketing-metrics-to-prove-roi/)

1. Impression
2. Reach
3. Website Traffic
4. Brand Lift
5. Avg. time on page
6. Bounce Rate
SmartAd provides BMO based a lightweight questionnaire served with every campaign to determine effects on upper-funnel-metrics like memorability and brand sentiment.

**The goal here is to design a reliable hypothesis testing algorithm for the `BIO` service, and determine whether a recent Ad campaign resulted in a significant lift in brand awareness.**

---

First we will experiment with classical frequnetist techniques, and then we move on to Machine Learning based approaches.

## MLOps Design

![image](https://user-images.githubusercontent.com/39389971/169132509-958aad9a-84da-40da-ab7a-5b1afecacc5c.png)
---

## Steps to work with this repo

1. clone the repo
2. create a new environment and install the `requirements.txt`
3. run `dvc pull` to get the dataset and model files
4. use the `Notebooks/logistic_regression.ipynb` notebook as an example for accessing data and training a model
5. use good names for the mlflow experiment name and run names. Experiment name should be prefixed with our names.(we can discuss this)

## Contributors

![Contributors list](https://contrib.rocks/image?repo=Hen0k/smartAd-abtesting)

Made with [contrib.rocks](https://contrib.rocks).
