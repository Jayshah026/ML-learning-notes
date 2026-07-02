# Way to train models

```Python 

- There are two ways to training a model

1. Batch Learning (Offline ML)
2. Online Learning (Incremental ML)

```
---

## Batch Learning 

```Python

- Batch learning = offline learning.

- We train the model on the full dataset we currently have, all at once (not gradually).

- When new data shows up, we don't update the existing model live — we retrain from scratch. 


Process : 

- Collect all available data up to that point.

- Train offline — the full model is trained on this entire dataset in one go (this can take hours/days depending on size, done on a separate training system, not in production).

- Deploy the trained model to production/servers. This deployed model is now static — it does not learn or update itself while serving predictions.

- Model serves predictions to users. Over time, its performance can start to degrade as the real world shifts (this is called model/data drift).

- When new data accumulates, you combine old + new data, retrain the entire model from scratch offline, and redeploy the new version — replacing the old one.

- Repeat this cycle periodically (e.g : 24 hours, weekly, monthly, etc..) — this whole process can be automated as an MLOps pipeline.

```
---

## Online Learning

```Python

- Initialize the model — usually starts with some base state (could be randomly initialized, or pre-trained on a small initial batch to give it a starting point).

- Data arrives as a stream — one instance, or a small mini-batch, at a time (not the whole dataset at once).
Model makes a prediction on that incoming data point (if applicable — e.g., predicting fraud/not-fraud for a transaction as it happens).

- Model learns immediately — it compares its prediction to the actual outcome (once known) and updates its internal parameters/weights slightly based on just that one data point/mini-batch.

- Discard the data point — since the model already learned from it, that specific data doesn't need to be stored or reprocessed. (This is what makes it memory-efficient — you never need the full dataset in memory at once.)

- Repeat continuously — steps 2–5 keep happening as new data keeps flowing in, forever, in real time. The model is never "done" training; it's always slightly evolving.

- Model stays live in production the entire time — there's no separate "train then deploy" split like batch learning. Training and serving happen in the same continuous loop.

```
