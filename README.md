# nyc-taxi-trip-duration

Due to time limitations in completing the assignment, I have chosen to
1) not use external datasets, and
2) use random forest regression instead of xgboost.

Regarding 1)
If I had more time, I would have possibly included a) fastest route distance estimates, and maybe b) weather/temperature data

Regarding 2)
I chose to use random forest regression because I was having some issues installing the xgboost python package. Rather than spending a few hours on troubleshooting the installation, I chose to just use random forest regression since the difference in performance does not seem to be too big (Source: http://fastml.com/what-is-better-gradient-boosted-trees-or-random-forest/). I also have no experience using xgboost so I wasn't sure how much time/work would be necessary in learning how to tune the hyper parameters.


Before starting the assignment, I viewed existing kernals and saw how other experienced kagglers were structuring their model. I gained a lot of insight from other models so my approach is not very original. I took bits and pieces from various other kagglers and applied what I thought made sense, in addition to making changes where I felt was necessary. Also, I was able to view a lot of the data exploration processes different kagglers took, so it saved me a lot of time from exploring on my own. As a result, I've decided to exclude a lot of the exploratory data process from my ipynb.


Below are the list of features used in my model.

DATETIME
- pickup_weekday
- pickup_hour
- pickup_weekday_hour

LOCATION
- pickup_longitude
- pickup_latitude
- dropoff_longitude
- dropoff_latitude
- passthru_latitude
- passthru_longitude
- pickup_cluster
- dropoff_cluster

DISTANCE
- haversine_distance
- manhattan_distance

AVG SPEED
- avg_pickup_weekday_haversine_speed
- avg_pickup_weekday_manhattan_speed
- avg_pickup_hour_haversine_speed
- avg_pickup_hour_manhattan_speed
- avg_pickup_weekday_hour_haversine_speed
- avg_pickup_weekday_hour_manhattan_speed
- avg_pickup_cluster_haversine_speed
- avg_pickup_cluster_manhattan_speed
- avg_dropoff_cluster_haversine_speed
- avg_dropoff_cluster_manhattan_speed
- avg_pickup_cluster_pickup_hour_haversine_speed
- avg_pickup_cluster_pickup_hour_manhattan_speed
- avg_dropoff_cluster_pickup_hour_haversine_speed
- avg_dropoff_cluster_pickup_hour_manhattan_speed
- avg_pickup_cluster_dropoff_cluster_haversine_speed
- avg_pickup_cluster_dropoff_cluster_manhattan_speed

OTHER
- passenger_count
- store_and_fwd_flag
