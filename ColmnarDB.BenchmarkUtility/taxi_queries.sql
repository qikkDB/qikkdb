SELECT cab_type, COUNT(*) FROM trips GROUP BY cab_type;
SELECT passenger_count, AVG(total_amount) FROM trips GROUP BY passenger_count;
SELECT passenger_count, YEAR(pickup_datetime) AS pickup_year, COUNT(*) FROM trips GROUP BY passenger_count, pickup_year;	 
SELECT passenger_count, YEAR(pickup_datetime) AS pickup_year, cast(trip_distance as int) AS distance, COUNT(*) AS the_count FROM trips GROUP BY passenger_count, pickup_year, distance ORDER BY pickup_year, the_count DESC;