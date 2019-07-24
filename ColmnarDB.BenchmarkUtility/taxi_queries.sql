SELECT cab_type, COUNT(cab_type) FROM trips GROUP BY cab_type;
SELECT passenger_count, AVG(total_amount) FROM trips GROUP BY passenger_count;
SELECT passenger_count, YEAR(pickup_datetime), COUNT(cab_type) FROM trips GROUP BY passenger_count, cab_type;