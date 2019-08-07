SELECT cab_type, COUNT(passenger_count) FROM trips GROUP BY cab_type;
SELECT passenger_count, AVG(total_amount) FROM trips GROUP BY passenger_count;
SELECT passenger_count, YEAR(pickup_datetime) AS pickup_year, COUNT(passenger_count) FROM trips GROUP BY passenger_count, pickup_year;
SELECT passenger_count, YEAR(pickup_datetime) AS pickup_year, CAST(trip_distance AS INT) AS distance, COUNT(passenger_count) AS the_count FROM trips GROUP BY passenger_count, pickup_year, distance ORDER BY pickup_year, the_count DESC;