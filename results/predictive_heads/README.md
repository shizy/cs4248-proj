# Predictive Heads Experiment Results

This document records the SQuAD v1.1 evaluation results for all predictive heads experiments.

---

## Baseline

### Baseline Linear Head
```json
{
  "exact": 81.16367076631977,
  "f1": 88.58053911242423,
  "total": 10570,
  "HasAns_exact": 81.16367076631977,
  "HasAns_f1": 88.58053911242423,
  "HasAns_total": 10570
}
```

---

## Without Differentiated Learning Rate

### Conditioned Span Head
```json
{
  "exact": 81.57994323557237,
  "f1": 88.62590078903628,
  "total": 10570,
  "HasAns_exact": 81.57994323557237,
  "HasAns_f1": 88.62590078903628,
  "HasAns_total": 10570
}
```

### Multi-Sample Dropout Head
```json
{
  "exact": 81.3434247871334,
  "f1": 88.54008636868967,
  "total": 10570,
  "HasAns_exact": 81.3434247871334,
  "HasAns_f1": 88.54008636868967,
  "HasAns_total": 10570
}
```

### Convolutional Boundary Head
```json
{
  "exact": 0.6622516556291391,
  "f1": 1.9178409403828263,
  "total": 10570,
  "HasAns_exact": 0.6622516556291391,
  "HasAns_f1": 1.9178409403828263,
  "HasAns_total": 10570
}
```

### Residual MLP Head
```json
{
  "exact": 81.28666035950805,
  "f1": 88.36290178119224,
  "total": 10570,
  "HasAns_exact": 81.28666035950805,
  "HasAns_f1": 88.36290178119224,
  "HasAns_total": 10570
}
```

---

## With Differentiated Learning Rate

### Conditioned Span Head
```json
{
  "exact": 81.42857142857143,
  "f1": 88.75472622133614,
  "total": 10570,
  "HasAns_exact": 81.42857142857143,
  "HasAns_f1": 88.75472622133614,
  "HasAns_total": 10570
}
```

### Convolutional Boundary Head with Higher Head LR
```json
{
  "exact": 0.6622516556291391,
  "f1": 1.9178409403828263,
  "total": 10570,
  "HasAns_exact": 0.6622516556291391,
  "HasAns_f1": 1.9178409403828263,
  "HasAns_total": 10570
}
```

### Conditioned Span Head - 3 Epochs
```json
{
  "exact": 81.06906338694418,
  "f1": 88.66767552326267,
  "total": 10570,
  "HasAns_exact": 81.06906338694418,
  "HasAns_f1": 88.66767552326267,
  "HasAns_total": 10570
}
```

### Conditioned Span Head - Cased Backbone, 3 Epochs
```json
{
  "exact": 81.48533585619678,
  "f1": 88.90065945683365,
  "total": 10570,
  "HasAns_exact": 81.48533585619678,
  "HasAns_f1": 88.90065945683365,
  "HasAns_total": 10570
}
```

### Residual MLP Head with Differentiated LR
```json
{
  "exact": 81.25827814569537,
  "f1": 88.5920050731808,
  "total": 10570,
  "HasAns_exact": 81.25827814569537,
  "HasAns_f1": 88.5920050731808,
  "HasAns_total": 10570
}
```

### Depthwise Convolutional Head with ReZero
```json
{
  "exact": 0.6622516556291391,
  "f1": 1.9178409403828263,
  "total": 10570,
  "HasAns_exact": 0.6622516556291391,
  "HasAns_f1": 1.9178409403828263,
  "HasAns_total": 10570
}
```

### Logit-Space Boundary Convolution Head
```json
{
  "exact": 81.05960264900662,
  "f1": 88.5519893162409,
  "total": 10570,
  "HasAns_exact": 81.05960264900662,
  "HasAns_f1": 88.5519893162409,
  "HasAns_total": 10570
}
```
