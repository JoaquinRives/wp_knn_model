=> Water permeability prediction model

- Description:
k nearest neighbor regression model for prediction of the water permeability
level of forest soil. The model is intended to be use in the forest industry to
guide routing decisions in harvesting operations.

The soil water permeability 𝑥𝑤𝑝 can be used as an indicator for the bearing capacity 
of the soil, which is a crucially important factor in harvest operations with heavy machinery.

The predicted value of 𝑥𝑤𝑝 will be used to guide routing decisions in order to minimize
risks  for the forest harvester.

A route consists from a fixed number of spatially distributed points, for all of which 
we need a prediction on the 𝑥𝑤𝑝 level, in order to evaluate the route’s goodness.

The predictions are made using the k Nearest Neighbors Algorithm with Euclidean distance.

# To install from github repo:
>>> pip install git+git://github.com/JoaquinRives/wp_knn_model

# To install locally:
>>> pip install -e /wp_forest_project_public/packages/wp_knn_model





