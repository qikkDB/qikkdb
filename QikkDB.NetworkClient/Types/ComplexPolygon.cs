using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text.RegularExpressions;
using Google.Protobuf.Collections;

namespace QikkDB.Types
{
    /// <summary>
    /// The class representing complex polygon object which consists of one or more polygons.
    /// </summary>
    public partial class ComplexPolygon
    {
        public const int MAX_POLYGONS_NUMBER = 8; //TODO probably would be better in config file or some constants fil

        /// <summary>
        /// Constructor for creating complex polygon and initializing.
        /// </summary>
        /// <param name="wktPolygon">String of well known text formatted polygon.</param> 
        /// <param name="spaceBetweenItems">Represents if there is a space between geo points after a comma and between
        /// polygons also after a comma. Default value is set to 'false'.</param>
        /// <exception cref="FormatException">Format exception with a message that explains the reason why the
        /// exception have been thrown.</exception>
        public ComplexPolygon(string wktPolygon)
        {
            this.polygons_ = new RepeatedField<Polygon>();
            string[] polygons;

            polygons = wktPolygon.Replace(", ", ",").Split(new string[]{ "),("}, StringSplitOptions.RemoveEmptyEntries);

            // regular expression that ensures correct format (according to well known texts) of geo points of which
            // a polygon consists
            var regex = new Regex("((-?[0-9]+(\\.[0-9]+)?) (-?[0-9]+(\\.[0-9]+)?)(, ?)?)+");

            polygons[0] = polygons[0].Replace("POLYGON((", "").Replace("POLYGON ((", ""); //remove prefix
            polygons[polygons.Length - 1] = polygons[polygons.Length - 1].Replace("))", ""); //remove suffix

            if (polygons.Length > MAX_POLYGONS_NUMBER)
            {
                throw new FormatException(
                    "Well known text polygon is in wrong format - encountered more polygons than " +
                    "is max number of polygons in complex polygon. [" + wktPolygon + "]");
            }

            for (var i = 0; i < polygons.Length; i++)
            {
                if (polygons[i].Contains(')') || polygons[i].Contains('('))
                {
                    throw new FormatException(
                        "Well known text polygon is in wrong format - wrong number of '(' or ')'.");
                }

                if (!regex.IsMatch(polygons[i]))
                {
                    throw new FormatException("Well known text polygon is in wrong format - there is a wrong " +
                                              "character that is not a number, comma nor space. [" + wktPolygon + "]");
                }

                string[] points;

                //points = polygons[i].Replace(", ", ",").Split(",");
                points = polygons[i].Split(',');

                var geoPoints = new GeoPoint[points.Length];
                for (var j = 0; j < points.Length; j++)
                {
                    var coordinates = points[j].Split(' ');

                    if (coordinates.Length > 2)
                    {
                        throw new FormatException(
                            "Well known text polygon is in wrong format - wrong number of coordinates, there is more of them per geo point. [" +
                            wktPolygon + "]");
                    }

                    try
                    {
                        geoPoints[j] = new GeoPoint();
                        geoPoints[j].Latitude = float.Parse(coordinates[0], CultureInfo.InvariantCulture);
                        geoPoints[j].Longitude = float.Parse(coordinates[1], CultureInfo.InvariantCulture);
                    }
                    catch (IndexOutOfRangeException)
                    {
                        throw new FormatException("Well known text polygon is in wrong format - encountered less " +
                                                  "coordinates than needed to parse geo point. [" + wktPolygon + "]");
                    }
                    catch (FormatException)
                    {
                        throw new FormatException("Well known text polygon is in wrong format - encountered problem " +
                                                  "with geo point coordinates. [" + wktPolygon + "]");
                    }
                }

                //test first and last geo point correctness:
                if (Math.Abs(geoPoints[0].Latitude - geoPoints[points.Length - 1].Latitude) >= 0.0001f ||
                    Math.Abs(geoPoints[0].Longitude - geoPoints[points.Length - 1].Longitude) >= 0.0001f)
                {
                    throw new FormatException("Well known text polygon is in wrong format - the first geo point must " +
                                              "be the same as the last one [" + wktPolygon + "]");
                }

                var polygon = new Polygon(geoPoints);

                Polygons.Add(polygon);
            }
        }

        /// <summary>
        /// Getter for polygons of a complex polygon.
        /// </summary>
        /// <returns>List of polygons.</returns>
        public List<Polygon> GetPolygons()
        {
            return Polygons.ToList();
        }

        /// <summary>
        /// Setter for setting polygons to a complex polygon.
        /// </summary>
        /// <param name="polygons">List of polygons.</param>
        public void SetPolygons(List<Polygon> polygons)
        {
            Polygons.Clear();
            Polygons.AddRange(polygons);
        }

        /// <summary>
        /// Method for adding one polygon to a list of polygons of this particular complex polygon.
        /// </summary>
        /// <param name="polygon">Polygon that will be added.</param>
        public void AddPolygon(Polygon polygon)
        {
            Polygons.Add(polygon);
        }

        /// <summary>
        /// Converts polygons to GPU representation
        /// </summary>
        /// <param name="polygons">Polygons to convert</param>
        /// <returns>Tuple of array for the GPU</returns>
        public static (GeoPointNative[] points, Int32[] pointsStartIdx, Int32[] pointsCount, Int32[] polysStartIdx, Int32[]
            polysCount) PrepareGpuPolygon(ComplexPolygon[] polygons)
        {
            // Points of polygons
            List<GeoPointNative> polyPoints = new List<GeoPointNative>();
            // Start indexes of each polygon in point array
            List<Int32> pointIdx = new List<int>();
            // Number of points of each polygon
            List<Int32> pointCount = new List<int>();
            // Start indexes of each complex polygon in polygon array
            List<Int32> polyIdx = new List<int>();
            // Number of polygons of each complex polygon
            List<Int32> polyCount = new List<int>();
            // ToDo: Mozno paralelizovat mimo testov
            foreach (var cp in polygons)
            {
                var subPolygons = cp.GetPolygons();
                polyIdx.Add(pointIdx.Count);
                polyCount.Add(subPolygons.Count);
                foreach (var polygon in subPolygons)
                {
                    var points = polygon.GetGeoPoints();
                    pointIdx.Add(polyPoints.Count);
                    pointCount.Add(points.Length + 2);
                    // Necessary for the raycasting to work, separates components of complex polygons
                    polyPoints.Add(new GeoPointNative() {latitude = 0, longitude = 0});
                    var nativePoints = from point in points
                        select new GeoPointNative() {latitude = point.Latitude, longitude = point.Longitude};
                    polyPoints.AddRange(nativePoints);
                    polyPoints.Add(new GeoPointNative() {latitude = 0, longitude = 0});
                }
            }

            return (polyPoints.ToArray(), pointIdx.ToArray(), pointCount.ToArray(), polyIdx.ToArray(),
                polyCount.ToArray());
        }

        /// <summary>
        /// Method that converts class to a string representation.
        /// </summary>
        /// <returns>ComplexPolygon in format of well known text.</returns>
        public string ToWktString()
        {
            var wkt = "POLYGON(";

            for (var i = 0; i < Polygons.Count; i++)
            {
                wkt += "(";

                var geoPoints = Polygons[i].GetGeoPoints();
                for (var j = 0; j < geoPoints.Length; j++)
                {
                    wkt += geoPoints[j].Latitude.ToString(CultureInfo.InvariantCulture) + " " +
                           geoPoints[j].Longitude.ToString(CultureInfo.InvariantCulture);

                    if (j != geoPoints.Length - 1)
                    {
                        wkt += ",";
                    }
                }

                wkt += ")";

                if (i != Polygons.Count - 1)
                {
                    wkt += ",";
                }
            }

            wkt += ")";

            return wkt;
        }
        
        public static bool operator ==(ComplexPolygon lhs, ComplexPolygon rhs)
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

        public static bool operator !=(ComplexPolygon lhs, ComplexPolygon rhs)
        {
            return !(lhs == rhs);
        }
    }
}