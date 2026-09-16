# Pocket Perceptron

A MATLAB / GNU Octave implementation of the **pocket algorithm** (Gallant, 1990): a
single-layer perceptron that keeps, "in its pocket", the weight vector that classified
the largest number of training patterns correctly.

Written for the Computational Intelligence lab, Department of Informatics and
Telecommunications, University of Ioannina (January 2025), and cleaned up for
publication.

## Why the pocket algorithm exists

The perceptron learning rule terminates only when the two classes are linearly
separable. When they are not, the weight vector oscillates forever, and whatever you
happen to be holding when you stop the loop can be arbitrarily bad — the last update is
not the best one.

The pocket algorithm adds a single idea: after every weight update, count how many of
the *N* training patterns the new weight vector classifies correctly, and store that
vector only if the count beats the best seen so far. What you keep is the pocket vector,
not the final one.

## Usage

```matlab
% from the repository root
data = load('data/nonseparable2D.csv');   % N-by-3: [x1 x2 target], target in {-1, +1}
[w, correct] = pocket_perceptron(data);   % w = [w1; w2; bias]
```

Optional arguments are the learning rate, the maximum number of epochs and whether to
plot:

```matlab
[w, correct] = pocket_perceptron(data, 0.01, 1000, true);
```

Run both demo datasets at once:

```matlab
>> demo
```

The figure shows the two classes, the decision boundary of every epoch as a faded trail,
and the final pocket boundary in red.

### Using the `mydata2D.mat` format

If your data comes as a patterns matrix `p` (2-by-N) and a targets row `t` (1-by-N):

```matlab
load('mydata2D.mat', 'p', 't');
pocket_perceptron([p' t']);
```

## Termination condition

The lab assignment asks for the exact condition under which the loop stops. It is **not**
"no weight changed in this epoch" and it is not a threshold on the error: the loop stops
as soon as the *pocket* vector classifies all *N* patterns correctly, i.e. when a full
linear separation has been found. If the classes are not linearly separable that never
happens, so the loop is bounded by `max_epochs` — the pocket algorithm has no natural
stopping point in that case, which is exactly why it keeps a best-so-far vector.

## Files

| Path | What it is |
| --- | --- |
| `pocket_perceptron.m` | The algorithm, with plotting of every intermediate boundary |
| `demo.m` | Runs the algorithm on both datasets |
| `data/separable2D.csv` | 60 patterns, two Gaussian clusters, linearly separable |
| `data/nonseparable2D.csv` | 80 patterns, overlapping clusters plus 4 flipped labels, **not** separable |
| `tools/make_datasets.py` | Regenerates both CSVs from a fixed seed and verifies separability with an LP feasibility test |
| `docs/pocket-perceptron-report-gr.pdf` | The original lab report (in Greek) |

## Note on the originally submitted version

The version handed in for the lab (preserved in this repository's first commit as
`pocketv2.m`) implemented the pocket criterion incorrectly: it compared a *running*
count of correct predictions accumulated part-way through an epoch — while the weights
were still being updated inside that same loop — instead of evaluating a fixed weight
vector against the whole training set.

On separable data this goes unnoticed. On the non-separable dataset in this repository
it does not:

| | reported score | patterns the stored weights actually classify correctly |
| --- | --- | --- |
| original submission | 61 / 80 | 58 / 80 |
| this version | 68 / 80 | 68 / 80 |

The stored weights were worse than the ones a correct implementation finds, and the
number printed at the end did not describe them. Fixed here; the Greek report in `docs/`
documents the original submission, so its code listing differs from the current source.

## Reference

S. I. Gallant, "Perceptron-based learning algorithms", *IEEE Transactions on Neural
Networks*, vol. 1, no. 2, pp. 179–191, 1990.

## License

[MIT](LICENSE)
