using System;
using System.Runtime.InteropServices;

namespace QikkDB
{
    /// <summary>
    /// Struct holding coordinates of a geo point
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct GeoPointNative : IEquatable<GeoPointNative>
    {
        public float latitude, longitude;

        /// <summary>
        /// Check if two GeoPoints represent the same point
        /// </summary>
        /// <param name="other">Point to compare</param>
        /// <returns>Comparison result</returns>
        public bool Equals(GeoPointNative other)
        {
            return Math.Abs(latitude - other.latitude) < 0.0001f && Math.Abs(longitude - longitude) < 0.0001f;
        }

        /// <summary>
        /// Check if two objects have the same value
        /// </summary>
        /// <param name="obj">Object to compare</param>
        /// <returns>Comparison result</returns>
        public override bool Equals(object obj)
        {
            if (ReferenceEquals(null, obj)) return false;
            return obj is GeoPointNative other && Equals(other);
        }

        /// <summary>
        /// Get Hash code of this object
        /// </summary>
        /// <returns>Hash code of this object</returns>
        public override int GetHashCode()
        {
            unchecked
            {
                return (latitude.GetHashCode() * 397) ^ longitude.GetHashCode();
            }
        }
        
        public static bool operator ==(GeoPointNative lhs, GeoPointNative rhs)
        {
            // Equals handles case of null on right side.
            return lhs.Equals(rhs);
        }

        public static bool operator !=(GeoPointNative lhs, GeoPointNative rhs)
        {
            return !(lhs == rhs);
        }
    }
}