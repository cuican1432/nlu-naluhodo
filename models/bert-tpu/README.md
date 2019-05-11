# Multi-labels Classification base on BERT

multi-labels classification task


### train

In train phase, we need to change model structure as output layer, and multi-classes classifier put softmax layer after as output layer.

Regarding multi-labels, softmax change to sigmoid layer, and loss change to sigmoid_cross_entropy_with_logits,
you can find it in `run_multilabels_classifier` `create_model()` function

### eval

Eval metric change to auc of every class, tf.metrics.auc used in `run_multilabels_classifier`  `metric_fn()`


### reference

Github Repo is referenced from https://github.com/yajian/bert