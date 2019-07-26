SELECT cab_type, COUNT(passenger_count) FROM trips GROUP BY cab_type;
SELECT passenger_count, AVG(total_amount) FROM trips GROUP BY passenger_count;
SELECT passenger_count, YEAR(pickup_datetime), COUNT(passenger_count) FROM trips GROUP BY passenger_count, YEAR(pickup_datetime);	 
SELECT passenger_count, YEAR(pickup_datetime), cast(trip_distance as int), COUNT(passenger_count) FROM trips GROUP BY passenger_count, YEAR(pickup_datetime), cast(trip_distance as int) ORDER BY YEAR(pickup_datetime), COUNT(passenger_count) DESC;