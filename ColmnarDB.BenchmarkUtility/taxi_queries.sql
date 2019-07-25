SELECT COUNT(passenger_count) FROM trips;
SELECT passenger_count, YEAR(pickup_datetime), COUNT(cab_type) FROM trips GROUP BY passenger_count, cab_type;