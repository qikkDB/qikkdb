using System;
using System.Globalization;

namespace ColmnarDB.Types
{
    /// <summary>
    /// The wrapper class for struct geo point.
    /// </summary>
    public partial class Point
    {
      
        /// <summary>
        /// Creates a geo point from well known text - point.
        /// </summary>
        /// <param name="wktPoint">String formatted according to well known texts - point.</param>
        public Point(string wktPoint)
        {
            var strPoint = wktPoint;
            strPoint = strPoint.Replace("POINT(", "").Replace("POINT (", ""); // remove prefix
            strPoint = strPoint.Replace(")", ""); //remove suffix

            var pointCoordinates = strPoint.Split(' ');

            if (pointCoordinates.Length != 2)
            {
                throw new FormatException(
                    "Well known text polygon is in wrong format - wrong number of coordinates. [" + wktPoint + "]");
            }

            GeoPoint = new GeoPoint
            {
                Latitude = float.Parse(pointCoordinates[0], CultureInfo.InvariantCulture),
                Longitude = float.Parse(pointCoordinates[1], CultureInfo.InvariantCulture)
            };
        }

        /// <summary>
        /// Creates a geo point from float coordianates.
        /// </summary>
        /// <param name="lat">Latitude</param>
        /// <param name="lon">Longitude</param>
        public Point(float lat, float lon)
        {
            GeoPoint = new GeoPoint
            {
                Latitude = lat,
                Longitude = lon
            };
        }

        /// <summary>
        /// Getter for geo point.
        /// </summary>
        /// <returns>Geo point with coordinates.</returns>
        public GeoPoint GetGeoPoint()
        {
            return GeoPoint;
        }

        /// <summary>
        /// Sets a new geo point.
        /// </summary>
        /// <param name="geoPoint">Geo point that will be set.</param>
        public void SetGeoPoint(GeoPoint geoPoint)
        {
            GeoPoint = geoPoint;
        }

        /// <summary>
        /// Method that converts class to a string representation.
        /// </summary>
        /// <returns>Point in format of well known text.</returns>
        public string ToWktString()
        {
            var wkt = "POINT(";

            wkt += GeoPoint.Latitude.ToString(CultureInfo.InvariantCulture) + " " +
                   GeoPoint.Longitude.ToString(CultureInfo.InvariantCulture);

            wkt += ")";

            return wkt;
        }
        
        public static bool operator ==(Point lhs, Point rhs)
        {
            // Check for null on left side.
            if (ReferenceEquals(lhs, null))
            {
                if (ReferenceEquals(rhs, null))
                {
                    // null == null = true.
                    return true;
                }

                // Only the left side is null.
                return false;
            }
            // Equals handles case of null on right side.
            return lhs.Equals(rhs);
        }

        public static bool operator !=(Point lhs, Point rhs)
        {
            return !(lhs == rhs);
        }
        
    }
}