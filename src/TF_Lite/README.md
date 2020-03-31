# Create TensorFlow Lite models

First, freeze the graph of pre-trained models (this is necessary since TensorFlow 1.11 does not allow to save frozen graph; use a newer version of TensorFlow as 1.15).
Use the script `freeze_graph.py` in the following manner:

```script
python freeze_graph.py --model_dir ./data/lenet5/mnist/ --output_node_names logits
```

Then, you can convert this graph to a TF Lite model, as `lenet5_int_quant.py` does, for example.
