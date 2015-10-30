Part 1
=

1. Describe your methodology. Why did you pick it?
     * I dropped all rows without the variable of interest, closing price. An interesting avenue of future exploration might be to build a classifier to distinguish the houses that close from those that don't. This could then motivate a feature in our original task that might provide some evidence that the listing price is too high. I didn't pursue this due to time constraints.
    *  I also dropped any rows with null values in any of the columns. Time constraints were the principal motivator for this decision.
    *  I encoded the categorical variables using a one hot encoding. One reason was to express the fact that the raw 'Pool' variable encoded the binary relationship of two variables ('Communal Pool' and 'Private Pool').
     * I did a minor bit of feature engineering on the number of bathrooms and closing date variables. An increased time on the market suggests evidence of too high a listing price, or, perhaps, lower inherent value of the property. I thought conditioning on the year and month might encode some form of seasonality if it exists in the market. I also just guessed that it might be important to treat half bathrooms and full ones as separate variables.
    * The model used is a random forest regressor. I chose this model because of the minimal amount of data preparation required. For example, it handles feature scaling "out-of-the-box" and often avoids overfitting.
2. How well would you do if you excluded the list price?
    The model does rather poorly if you exclude the list price. Including it
improves performance by more than 5X.
3. What is the performance of your model? What error metrics did you choose?
    I chose mean absolute error (MAE) as the error metric. I chose this because it
highly interpretable. However, I doubt that it is an appropriate loss function
for the task in the real world. One could imagine a more appropriate loss
function penalizing overestimations of a price more than underestimation.
Motivating this would be the desire to minimize the amount of houses acquired
above the potential resale price. We could also consider the problem of
estimating the potential return on investment of the house. It might then follow
that mistakes in either direction could be treated as equally risky, and our
goal could be to minimize risk. With this as an impetus, it might be more
reasonable to have an asymmetric loss function that penalizes differences in
direction of loss more heavily (quadratically, for example) than errors of the
proper sign (linear loss in this direction is an example). The incredibly naive
baseline of predicting the mean price for the dataset has a MAE of around $113K,
compared to $7K for a model built using the list price and one of $36K without.
```
baseline             $113,649.90
with list price        $7,041.92
without list price    $36,613.40
```
4. How would you improve your model?
	One way would be to explore the text descriptions of the houses. It's
	possible that these contain additional features about the location or
	property itself that are not manifest in our structured data. If this
	approach is taken, it might make sense to treat this as an information
	extraction problem and attempt to infer additional structured data about the
	property. Other areas for exploration include learning a distinct model
per dwelling type, more feature engineering for latitude and longitude
(Haversine distance or perhaps zip codes might be interesting), and direct
optimization of the alternative loss functions mentioned.
5. How would you host your model in a production environment to predict values
   of homes
	I will assume we want to solve a problem similar to Opendoor's home
	page, where a potential seller might enter in a descriptive set of data about
	his or her home. One might perform the initial feature extraction in the
	app layer (or via a separate feature extraction service) and then ship the
	feature vectors to a prediction service whose responsibility would be to output
	predictions. This service might be as simple as a simple `Flask` app around
	`scikit-learn`'s models exposing a simple set of APIs (a la, `PREDICT
	<feature_vector>` `REGISTER <model>`. A nice example of this is Openscoring and
	AirBnB's discussion on "Architecting a Machine Learning System for
	Risk." The key insight into this system, is allowing for the deployment of
	models via some DSL, rather than requiring a code push/redeploy.
		

Part 2
=

Below are the top-10 trigrams from the `n-gram` Wikipedia article.

```
	n gram models
	an n gram
	syntactic n grams
	natural language processing
	out of vocabulary
	serve as the
	of n grams
	n gram model
	of vocabulary words
	computational linguistics and
```

Part 3
=

This implementation has amortized `O(1)` lookup, insertion, and deletion since it uses the elementary operations of an ordered dictionary which has the same complexity.  I would likely use `functools.lru_cache` for a production implementation. This implementation does not handle concurrent operations.

Running the Code
=
```
$ pip3 install -r requirements.txt
$ python3 part1.py --help # displays help
$ python3 part1.py data.csv # outputs CV scores
$ python3 part2.py # outputs top-10 ngrams
$ python3 part3.py -v # runs doctests
```

