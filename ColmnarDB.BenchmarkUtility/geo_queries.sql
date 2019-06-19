SELECT points FROM GeoPoint WHERE GEO_CONTAINS(POLYGON((17.0779 48.1303, 17.0912 48.1303, 17.0912 48.1391, 17.0779 48.1391, 17.0779 48.1303)), points);
SELECT points FROM GeoPoint WHERE GEO_CONTAINS(POLYGON((17.0867 48.1405, 17.0982 48.1405, 17.0982 48.1579, 17.0867 48.1579, 17.0867 48.1405)), points);
SELECT points FROM GeoPoint WHERE GEO_CONTAINS(POLYGON((17.0941 48.1439, 17.108 48.1439, 17.108 48.1557, 17.0941 48.1557, 17.0941 48.1439)), points);
SELECT points FROM GeoPoint WHERE GEO_CONTAINS(POLYGON((17.1008 48.1451, 17.1175 48.1451, 17.1175 48.1556, 17.1008 48.1556, 17.1008 48.1451)), points);
SELECT points FROM GeoPoint WHERE GEO_CONTAINS(POLYGON((17.1077 48.1528, 17.1204 48.1528, 17.1204 48.1682, 17.1077 48.1682, 17.1077 48.1528)), points);
SELECT points FROM GeoPoint WHERE GEO_CONTAINS(POLYGON((17.1114 48.1549, 17.1232 48.1549, 17.1232 48.1706, 17.1114 48.1706, 17.1114 48.1549)), points);
SELECT points FROM GeoPoint WHERE GEO_CONTAINS(POLYGON((17.1201 48.1649, 17.1356 48.1649, 17.1356 48.1798, 17.1201 48.1798, 17.1201 48.1649)), points);
SELECT points FROM GeoPoint WHERE GEO_CONTAINS(POLYGON((17.1259 48.1711, 17.1415 48.1711, 17.1415 48.1822, 17.1259 48.1822, 17.1259 48.1711)), points);
SELECT points FROM GeoPoint WHERE GEO_CONTAINS(POLYGON((17.1287 48.1808, 17.1444 48.1808, 17.1444 48.198, 17.1287 48.198, 17.1287 48.1808)), points);
SELECT points FROM GeoPoint WHERE GEO_CONTAINS(POLYGON((17.1373 48.1878, 17.1514 48.1878, 17.1514 48.2076, 17.1373 48.2076, 17.1373 48.1878)), points);
SELECT colID FROM TableA WHERE GEO_INTERSECT(colPolygon1, colPolygon2);
SELECT colID FROM TableA WHERE GEO_CONTAINS(POINT(15 10.9), colPolygon1);
SELECT colID FROM TableA WHERE GEO_CONTAINS(POINT(3 2), colPolygon1);
SELECT colID FROM TableA WHERE GEO_UNION(colPolygon1, colPolygon2);