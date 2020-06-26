using System;
using System.Linq;
using Google.Protobuf.Collections;

namespace QikkDB.Types
{
    /// <summary>
    /// Class representing a simple polygon that consists of geo points.
    /// </summary>
    public partial class Polygon
    {
        public const int MAX_POLYGON_INDICES = 32; //TODO probably would be better in config file or some constants file
        
        /// <summary>
        /// Constructor for creating a polygon with specified geo points.
        /// </summary>
        /// <param name="points">Array of geo points fof which the polygon consists.</param>
        public Polygon(GeoPoint[] points)
        {
            if (points.Length > MAX_POLYGON_INDICES)
            {
                throw new ArgumentException("Too many geopoints for polygon");
            }
            geoPoints_ = new RepeatedField<GeoPoint>();
            GeoPoints.AddRange(points);
        }

        /// <summary>
        /// Getter for getting geo points of a polygon.
        /// </summary>
        /// <returns>Array of geo points.</returns>
        public GeoPoint[] GetGeoPoints()
        {
            return GeoPoints.ToArray();
        }

        /// <summary>
        /// Setter for setting geo points to a polygon.
        /// </summary>
        /// <param name="points">Array of geo points.</param>
        public void SetGeoPoints(GeoPoint[] points)
        {
            GeoPoints.Clear();
            GeoPoints.AddRange(points);
        }

        /// <summary>
        /// Setter for setting one geo point at specified index.
        /// </summary>
        /// <param name="point">Geo point that will be set.</param>
        /// <param name="index">Index at which the geo point will be set.</param>
        public void SetGeoPointAtIndex(GeoPoint point, byte index)
        {
            GeoPoints[index] = point;
        }
        
        public static bool operator ==(Polygon lhs, Polygon rhs)
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

        public static bool operator !=(Polygon lhs, Polygon rhs)
        {
            return !(lhs == rhs);
        }
    }
}