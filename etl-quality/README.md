# ETL Data Quality Monitoring

Prototype accompanying a planned review of rule-based and unsupervised anomaly detection for ETL pipelines. The experiment illustrates how the two approaches behave on known faults in synthetic transaction data.

## Run

Use Python 3.11 or 3.12 in a virtual environment.

```sh
python -m pip install -r requirements.txt
python experiment.py --output results.csv
```

The script prints mean scores and writes one row per method and seed to CSV.

## Experiment

Each of five fixed seeds creates 4,000 clean training records and 1,600 independent test records. The script injects missing prices, invalid quantities, and unusually low transaction amounts into disjoint sets of 80 test records per fault type. The last fault stays within the individual field rules. The model uses quantity, price, amount, and the ratio between reported and expected amount. Its contamination setting is fixed at 2% on the training data. The hybrid flags a record if either detector flags it.

The CSV records precision, recall, F1, false-positive rate, and recall by fault type. The synthetic scenario assumes clean training records and a known relationship between transaction fields. Results depend on the injected faults, model features, and threshold. Use these results as an illustrative experiment alongside the literature review.

Related MBA project: *AI-Powered Data Quality Monitoring and Anomaly Detection for Enterprise Data Pipelines*.
