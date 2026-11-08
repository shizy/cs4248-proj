# Predictive Heads Experiment Results

This document records the SQuAD v1.1 evaluation results for all predictive heads experiments.

---

## Baseline

### Baseline Linear Head
Prediction:
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

### Baseline Linear Head - 3 Epochs
Prediction:
```json
{
  "exact": 81.05014191106906,
  "f1": 88.62980129145953,
  "total": 10570,
  "HasAns_exact": 81.05014191106906,
  "HasAns_f1": 88.62980129145953,
  "HasAns_total": 10570
}
```
Bootstrap test:
```json
  "p_values": {
    "exact_match": {
      "p_value": 0.7,
      "num_sig": 700,
      "total": 1000
    },
    "f1_score": {
      "p_value": 0.401,
      "num_sig": 401,
      "total": 1000
    }
  }
```

### Baseline Linear Head - Cased Backbone, 2 Epochs
Prediction:
```json
{
  "exact": 81.77861873226112,
  "f1": 88.81592696101092,
  "total": 10570,
  "HasAns_exact": 81.77861873226112,
  "HasAns_f1": 88.81592696101092,
  "HasAns_total": 10570
}
```
Bootstrap test:
```json
  "p_values": {
    "exact_match": {
      "p_value": 0.043,
      "num_sig": 43,
      "total": 1000
    },
    "f1_score": {
      "p_value": 0.187,
      "num_sig": 187,
      "total": 1000
    }
  }
```

### Baseline Linear Head - Cased Backbone, 3 Epochs
Prediction:
```json
{
  "exact": 81.30558183538317,
  "f1": 88.74024067472013,
  "total": 10570,
  "HasAns_exact": 81.30558183538317,
  "HasAns_f1": 88.74024067472013,
  "HasAns_total": 10570
}
```
Bootstrap test:
```json
  "p_values": {
    "exact_match": {
      "p_value": 0.364,
      "num_sig": 364,
      "total": 1000
    },
    "f1_score": {
      "p_value": 0.276,
      "num_sig": 276,
      "total": 1000
    }
  }
```

---

## Without Differentiated Learning Rate

### Conditioned Span Head
Prediction:
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
Bootstrap test:
```json
  "p_values": {
    "exact_match": {
      "p_value": 0.076,
      "num_sig": 76,
      "total": 1000
    },
    "f1_score": {
      "p_value": 0.424,
      "num_sig": 424,
      "total": 1000
    }
  }
```

### Multi-Sample Dropout Head
Prediction:
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
Bootstrap test:
```json
  "p_values": {
    "exact_match": {
      "p_value": 0.296,
      "num_sig": 296,
      "total": 1000
    },
    "f1_score": {
      "p_value": 0.607,
      "num_sig": 607,
      "total": 1000
    }
  }
```

### Convolutional Boundary Head
Prediction:
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
Bootstrap test:
```json
  "p_values": {
    "exact_match": {
      "p_value": 1.0,
      "num_sig": 1000,
      "total": 1000
    },
    "f1_score": {
      "p_value": 1.0,
      "num_sig": 1000,
      "total": 1000
    }
  }
```

### Residual MLP Head
Prediction:
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
Bootstrap test:
```json
  "p_values": {
    "exact_match": {
      "p_value": 0.348,
      "num_sig": 348,
      "total": 1000
    },
    "f1_score": {
      "p_value": 0.892,
      "num_sig": 892,
      "total": 1000
    }
  }
```

### Multi-Sample Dropout Head - 3 Epochs
Prediction:
```json
{
  "exact": 80.96499526963103,
  "f1": 88.39723274603118,
  "total": 10570,
  "HasAns_exact": 80.96499526963103,
  "HasAns_f1": 88.39723274603118,
  "HasAns_total": 10570
}
```
Bootstrap test:
```json
  "p_values": {
    "exact_match": {
      "p_value": 0.799,
      "num_sig": 799,
      "total": 1000
    },
    "f1_score": {
      "p_value": 0.846,
      "num_sig": 846,
      "total": 1000
    }
  }
```

### Logit-Space Boundary Convolution Head
Prediction:
```json
{
  "exact": 81.13528855250709,
  "f1": 88.51995391162374,
  "total": 10570,
  "HasAns_exact": 81.13528855250709,
  "HasAns_f1": 88.51995391162374,
  "HasAns_total": 10570
}
```
Bootstrap test:
```json
  "p_values": {
    "exact_match": {
      "p_value": 0.593,
      "num_sig": 593,
      "total": 1000
    },
    "f1_score": {
      "p_value": 0.688,
      "num_sig": 688,
      "total": 1000
    }
  }
```

---

## With Differentiated Learning Rate

### Conditioned Span Head
Prediction:
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
Bootstrap test:
```json
  "p_values": {
    "exact_match": {
      "p_value": 0.216,
      "num_sig": 216,
      "total": 1000
    },
    "f1_score": {
      "p_value": 0.207,
      "num_sig": 207,
      "total": 1000
    }
  }
```

### Convolutional Boundary Head with Higher Head LR
Prediction:
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
Bootstrap test:
```json
  "p_values": {
    "exact_match": {
      "p_value": 1.0,
      "num_sig": 1000,
      "total": 1000
    },
    "f1_score": {
      "p_value": 1.0,
      "num_sig": 1000,
      "total": 1000
    }
  }
```

### Conditioned Span Head - 3 Epochs
Prediction:
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
Bootstrap test:
```json
  "p_values": {
    "exact_match": {
      "p_value": 0.667,
      "num_sig": 667,
      "total": 1000
    },
    "f1_score": {
      "p_value": 0.355,
      "num_sig": 355,
      "total": 1000
    }
  }
```

### Conditioned Span Head - Cased Backbone, 3 Epochs
Prediction:
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
Bootstrap test:
```json
  "p_values": {
    "exact_match": {
      "p_value": 0.19,
      "num_sig": 190,
      "total": 1000
    },
    "f1_score": {
      "p_value": 0.102,
      "num_sig": 102,
      "total": 1000
    }
  }
```

### Residual MLP Head with Differentiated LR
Prediction:
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
Bootstrap test:
```json
  "p_values": {
    "exact_match": {
      "p_value": 0.392,
      "num_sig": 392,
      "total": 1000
    },
    "f1_score": {
      "p_value": 0.508,
      "num_sig": 508,
      "total": 1000
    }
  }
```

### Depthwise Convolutional Head with ReZero
Prediction:
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
Bootstrap test:
```json
  "p_values": {
    "exact_match": {
      "p_value": 1.0,
      "num_sig": 1000,
      "total": 1000
    },
    "f1_score": {
      "p_value": 1.0,
      "num_sig": 1000,
      "total": 1000
    }
  }
```

### Logit-Space Boundary Convolution Head
Prediction:
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
Bootstrap test:
```json
  "p_values": {
    "exact_match": {
      "p_value": 0.701,
      "num_sig": 701,
      "total": 1000
    },
    "f1_score": {
      "p_value": 0.591,
      "num_sig": 591,
      "total": 1000
    }
  }
```

### Multi-Sample Dropout Head with Differentiated LR
Prediction:
```json
{
  "exact": 81.2488174077578,
  "f1": 88.46088163964016,
  "total": 10570,
  "HasAns_exact": 81.2488174077578,
  "HasAns_f1": 88.46088163964016,
  "HasAns_total": 10570
}
```
Bootstrap test:
```json
  "p_values": {
    "exact_match": {
      "p_value": 0.388,
      "num_sig": 388,
      "total": 1000
    },
    "f1_score": {
      "p_value": 0.763,
      "num_sig": 763,
      "total": 1000
    }
  }
```

### Multi-Sample Dropout Head - Cased Backbone, 3 Epochs
Prediction:
```json
{
  "exact": 81.57994323557237,
  "f1": 88.96003906504971,
  "total": 10570,
  "HasAns_exact": 81.57994323557237,
  "HasAns_f1": 88.96003906504971,
  "HasAns_total": 10570
}
```
Bootstrap test:
```json
  "p_values": {
    "exact_match": {
      "p_value": 0.143,
      "num_sig": 143,
      "total": 1000
    },
    "f1_score": {
      "p_value": 0.071,
      "num_sig": 71,
      "total": 1000
    }
  }
```

### Residual MLP Head - Cased Backbone, 3 Epochs
Prediction:
```json
{
  "exact": 80.9271523178808,
  "f1": 88.68716287698824,
  "total": 10570,
  "HasAns_exact": 80.9271523178808,
  "HasAns_f1": 88.68716287698824,
  "HasAns_total": 10570
}
```
Bootstrap test:
```json
  "p_values": {
    "exact_match": {
      "p_value": 0.769,
      "num_sig": 769,
      "total": 1000
    },
    "f1_score": {
      "p_value": 0.345,
      "num_sig": 345,
      "total": 1000
    }
  }
```

### Logit-Space Boundary Convolution Head - 3 Epochs
Prediction:
```json
{
  "exact": 80.89877010406812,
  "f1": 88.47955054178212,
  "total": 10570,
  "HasAns_exact": 80.89877010406812,
  "HasAns_f1": 88.47955054178212,
  "HasAns_total": 10570
}
```
Bootstrap test:
```json
  "p_values": {
    "exact_match": {
      "p_value": 0.855,
      "num_sig": 855,
      "total": 1000
    },
    "f1_score": {
      "p_value": 0.721,
      "num_sig": 721,
      "total": 1000
    }
  }
```

### Logit-Space Boundary Convolution Head - Cased Backbone, 3 Epochs
Prediction:
```json
{
  "exact": 0.20813623462630085,
  "f1": 4.923762351344137,
  "total": 10570,
  "HasAns_exact": 0.20813623462630085,
  "HasAns_f1": 4.923762351344137,
  "HasAns_total": 10570
}
```
Bootstrap test:
```json
  "p_values": {
    "exact_match": {
      "p_value": 1.0,
      "num_sig": 1000,
      "total": 1000
    },
    "f1_score": {
      "p_value": 1.0,
      "num_sig": 1000,
      "total": 1000
    }
  }
```
