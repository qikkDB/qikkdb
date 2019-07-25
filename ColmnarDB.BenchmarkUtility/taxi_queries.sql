SELECT COUNT(passenger_count) FROM trips;
SELECT passenger_count, cab_type FROM trips GROUP BY passenger_count, cab_type;
SELECT YEAR(pickup_datetime) FROM trips