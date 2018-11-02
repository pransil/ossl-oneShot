# ossl-oneShot
Better version of 'One-Shot Self Supervised Learning'
For each new training sample:
  If within a current cluster, add to the cluster and adjust cluster center
  else create new cluster
  
Currently only one layer but in the future build multi-layer, one layer at a time, bottom up
  
Future work:
-  Instead of clusters, use embedded space where distance between items has meaning
-  Then can vary 'focus' param so that you get narrow / wide activation fns
-  Use resonance up and down the layers so that higher layer models affect info
    attended to in lower layers.
