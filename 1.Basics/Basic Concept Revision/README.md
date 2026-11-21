# Machine Learning Fundamentals

## AI, Machine Learning and Deep Learning

```python
- Artificial Intelligence (AI) is any technique that makes a machine behave in an intelligent way. The machine does not have to learn. A program where a human wrote every rule, like a chess engine or "if temperature > 30, switch on the fan", is AI.
- Machine Learning (ML) is the part of AI where the machine learns the rules from data instead of a human writing them. Traditional programming is rules plus data giving answers. ML is data plus answers giving rules.
- Deep Learning (DL) is the part of ML that uses neural networks with many layers. Its special strength is that it learns useful features by itself from raw input, such as pixels of an image. Classical ML usually needs a human to prepare the features first.
- DL needs much more data and computing power than classical ML. Face detection with a CNN is a typical DL task.
- Nesting: Deep Learning is inside Machine Learning, and Machine Learning is inside AI. All ML is AI, but not all AI is ML.
- How to tell them apart: a human wrote the rules means AI only. The rules were learned from data means ML. They were learned by a many-layered neural network from raw input means DL.
- The second part of the code below is a toy example of a rule learned from data. The cut-off 4.5 was calculated from the examples and was not typed by a human.

hours = 5
if hours >= 6:
    print("human rule: pass")
else:
    print("human rule: fail")

passed_hours = [6, 7, 8]
failed_hours = [1, 2, 3]
cut_off = (min(passed_hours) + max(failed_hours)) / 2
print(cut_off)
if hours > cut_off:
    print("learned rule: pass")
else:
    print("learned rule: fail")

OUTPUT :
human rule: fail
4.5
learned rule: pass
```

## Supervised Learning

```python
- Supervised learning is learning from data that contains both the inputs and the correct answer for each row. The correct answer is called the label or target.
- The model studies many examples of input and correct answer, and learns the mapping from one to the other. Later it gets only the input of a new row and predicts the answer.
- It is called supervised because the labels act like a teacher that tells the model what the right answer was.
- If the data has a correct-answer column, it is supervised learning.
- Examples: predicting whether a student passes, predicting a house price, deciding whether an email is spam.
- Supervised learning has two task types: classification and regression.

study_hours = [2, 5, 8, 3]
passed = ["no", "yes", "yes", "no"]

print(study_hours[1], passed[1])
print(len(study_hours), len(passed))

OUTPUT :
5 yes
4 4
```

## Unsupervised Learning

```python
- Unsupervised learning is learning from data that has no label column. There is no correct answer given to the model.
- The model only looks for structure in the data, such as which rows are similar to each other. It outputs group numbers, not names like "cat" or "dog".
- A human looks at the result afterwards and decides what each group means, for example "young customers".
- If the data has no correct-answer column, it is unsupervised learning.
- Unsupervised learning has two task types in this roadmap: clustering and association.
- Common mistake: saying an unsupervised model predicts the answer. There is no answer column to predict.

rows = [[4, 6], [30, 12], [5, 5], [25, 10]]

print(rows[0])
print(len(rows))
print(len(rows[0]))

OUTPUT :
[4, 6]
4
2
```

## Reinforcement Learning

```python
- Reinforcement learning is learning by trial and error with rewards. There is no fixed dataset.
- An agent is the learner, such as a game-playing program. The environment is the world it acts in, such as the game. An action is what the agent does, such as a move. A reward is the feedback it gets, a positive number for good results and a negative number for bad results.
- The data is created by the agent's own actions. The goal is to maximize the total reward over time, so a move that looks bad now can still be right if it leads to a win later.
- Example: a robot gets +1 for each step forward and a penalty when it falls.

rewards = [1, 1, -5, 1]
total = 0
for reward in rewards:
    total = total + reward
    print(total)

OUTPUT :
1
2
-3
-2
```

## Classification

```python
- Classification is a supervised task where the label is a category chosen from a set of pre-defined classes. The classes are fixed before training. The model puts each new row into one of them, based on its characteristics (its features).
- The label is a category, not a measured quantity. A value in between two classes has no meaning. There is no class 1.5 between cat and dog.
- Binary classification has two classes, such as spam and not spam. Multiclass classification has three or more, such as the three flower species in the Iris dataset.
- How it works underneath: most classifiers calculate a score or probability for each class and output the class with the highest one. For example 0.92 for spam and 0.08 for not spam gives spam.
- Labels written as numbers, like 0 and 1 for stayed and left, are still categories, so the task is still classification.
- Logistic Regression has regression in its name but it is a classification model.

classes = ["spam", "not spam"]
email_has_lottery = True
if email_has_lottery:
    prediction = classes[0]
else:
    prediction = classes[1]

print(prediction)
print(prediction in classes)

OUTPUT :
spam
True
```

## Regression

```python
- Regression is a supervised task where the label is a continuous number, and the model outputs a number.
- A quick test: if a value in between two labels makes sense, such as 72.4 marks, it is regression.
- The model learns how the features relate to the number, then calculates a value for a new row. The prediction will not be exact every time. The aim is to be close, and the distance from the true value is measured with error metrics such as MAE and RMSE (Topic 11).
- Examples: house price, marks, temperature.
- Ordinal labels like star ratings from 1 to 5 can be treated as classification or regression. The choice must be justified.
- The code below uses the model marks = 8 x hours + 20 only to show that the output is a number.

hours = 6
predicted_marks = 8 * hours + 20
print(predicted_marks)

hours = 6.5
predicted_marks = 8 * hours + 20
print(predicted_marks)

OUTPUT :
68
72.0
```

## Clustering

```python
- Clustering is an unsupervised task that groups similar rows together when no labels exist and no groups are defined in advance.
- The model gives each row a group number. A human decides what the groups mean, such as "young customers" and "senior customers".
- Underneath, K-Means keeps K centres. Each row joins the group of its nearest centre, and then each centre moves to the average of its group. This repeats until the groups stop changing. K-Means is covered in Topic 11.
- Examples: customer segments, grouping news articles by topic with no topics defined beforehand.
- The code below shows one step: assigning the age 22 to the nearer of two centres, 20 and 50.

age = 22
distance_0 = abs(age - 20)
distance_1 = abs(age - 50)
if distance_0 < distance_1:
    group = 0
else:
    group = 1

print(distance_0, distance_1)
print(group)

OUTPUT :
2 28
0
```

## Association

```python
- Association is an unsupervised task that finds items which often occur together in the same transaction, with no labels involved.
- It looks at combinations of items, not groups of rows. Clustering groups rows, while association finds item combinations.
- Example: market basket analysis, where many baskets contain both bread and butter.
- The simplest measure is counting how many transactions contain the items together. This roadmap only names association and does not go deeper.

baskets = [["bread", "butter"], ["bread", "milk"], ["bread", "butter", "milk"], ["milk"]]
count = 0
for basket in baskets:
    if "bread" in basket and "butter" in basket:
        count = count + 1

print(count)
print(len(baskets))

OUTPUT :
2
4
```

## Dataset, Sample, Feature and Target

```python
- A dataset is the whole table of data.
- A sample, also called a row, record, instance or example, is one real-world thing in the table, such as one house or one student.
- A column is one property measured for every sample.
- A feature is a column the model uses as input. All features together are called X.
- The target, also called the label, is the column the model must predict. It is called y.
- Shapes: X has shape (number of samples, number of features), and y has shape (number of samples,). Row i of X always matches y[i].
- The target must never be inside X, because the model would see the answer while training. This is called data leakage (Topic 7).
- ID columns like house_id are left out of the features because they identify a row but say nothing about the target.
- In unsupervised learning there is X but no y.

dataset = [[900, 2, 35], [1500, 3, 80], [1200, 3, 50]]
X = [[900, 2], [1500, 3], [1200, 3]]
y = [35, 80, 50]

print(len(dataset), len(dataset[0]))
print(len(X), len(X[0]))
print(len(y))
print(X[0], y[0])

OUTPUT :
3 3
3 2
3
[900, 2] 35
```

## Algorithm, Model, Training and Inference

```python
- An algorithm is a fixed recipe for learning from data. It exists before any data is seen. Linear regression is an algorithm.
- A model is the learned result: a template with its learned numbers filled in. These numbers are called parameters. For linear regression the template is price = w x area + b, and after training it might become price = 2 x area + 1.
- Training, also called fitting, runs the algorithm on data that has answers and adjusts the parameters to reduce its mistakes.
- Inference, also called prediction, uses the finished model on a new row. The parameters stay fixed and nothing is learned.
- Training changes the parameters. Inference only uses them.
- One algorithm trained on different data gives different models.
- In the code below, the parameters change from 0 and 0 to 2 and 1 to show the effect of training. Then the finished model predicts for a new area.

area = 1
w = 0
b = 0
print(w * area + b)

w = 2
b = 1
print(w * area + b)

new_area = 4
print(w * new_area + b)

OUTPUT :
0
3
9
```

## Generalization and Good Fit

```python
- Generalization is how well a model performs on data it has never seen. This is the real goal, because a model is only useful on new data.
- Training error is the model's mistakes on the data it learned from. Test error is its mistakes on unseen data.
- A good fit has low training error and low test error, with a small gap between them.
- A training score alone proves nothing, like a student who only practises questions they have already memorized. A model must be judged on unseen data.
- Cross-validation, bias and variance, and regularization are covered in Stages 1 to 3 of the main roadmap.

train_error = 3
test_error = 4
gap = test_error - train_error
print(gap)

OUTPUT :
1
```

## Overfitting

```python
- Overfitting means the model fits the training data extremely well, including its random noise, but performs badly on new data.
- Every dataset has a real pattern plus random noise. A good model learns the pattern. An overfit model is flexible enough to memorize the noise too, and noise does not repeat in new data.
- Signature: very low training error and much higher test error. The gap is the warning sign.
- It comes mostly from a model that is too flexible for the amount of data, not from too much training.
- Example: a student who memorizes 50 practice answers word for word and fails the new exam.

train_error = 2
test_error = 30
gap = test_error - train_error
print(gap)
if gap > 10:
    print("overfitting: big gap")

OUTPUT :
28
overfitting: big gap
```

## Underfitting

```python
- Underfitting means the model is too simple to capture the pattern in the data.
- Signature: high training error and high test error. It does badly on the data it learned from and on new data.
- Example: a student who studies for five minutes and learns one vague rule, then fails the practice questions and the exam.
- Fix: use a model that can learn more of the pattern. Overfitting is the opposite problem, so adding complexity helps only up to a point.

train_error = 18
test_error = 20
if train_error > 15 and test_error > 15:
    print("underfitting: both errors are high")

OUTPUT :
underfitting: both errors are high
```