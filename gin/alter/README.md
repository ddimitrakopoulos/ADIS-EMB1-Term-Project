Modified GIN Architecture
- Hardcoded datasets 
- The number of GIN layers is treated as a hyperparameter (`num_layers`)
- Node embeddings from all GIN layers are preserved
- Each layer’s node embeddings are then:
  - Globally pooled as the original did and finally concatenated to form the graph embedding
- The classifier is expanded to a two-layer MLP instead of a single linear layer
- MUTAG has no node_attributes, but has node_labels, so those are converted to one-hot vectors and are used as the initial node embeddings at the message passing algorithm. Basically, the atom type of each node becomes each node's feature.
- IMDB-MULTI has no node_labels or attributes, on kaggle one-hot encoded node degree vectors caused out-of-memory errors, so each node is represented using a single scalar feature equal to its degree. (acc is actually ugly to look at)
- ENZYMES has node_attributes, which are now being utilized compared to the other methods, so the acc is pretty much not chopped.
- kudos to [https://colab.research.google.com/drive/1b6SWugNKnxsI0L9auX1zwszlXf3rRZyS?usp=sharing#scrollTo=1YsYYubhTE22](https://colab.research.google.com/drive/1b6SWugNKnxsI0L9auX1zwszlXf3rRZyS?usp=sharing#scrollTo=1YsYYubhTE22)

Wake up in the morning feeling like P Diddy<br>
Grab my glasses, I'm out the door, I'm gonna hit this city<br>
Before I leave, brush my teeth with a bottle of GIN<br>
'Cause when I leave for the night, I ain't coming back

