using System;
using System.Collections.Generic;
using ColmnarDB.NetworkClient;

namespace ColmnarDB.BenchmarkUtility
{
    public class Program
    {
        public static readonly string IpAddress = "127.0.0.1";
        public static readonly short Port = 12345;

        /// <summary>
        /// Load benchmark queries from a file, execute them one by one and save results.
        /// </summary>
        public static void Main(string[] args)
        {
            ColumnarDBClient client = new ColumnarDBClient(IpAddress,Port);
            client.Connect();

            client.UseDatabase("TargetLoc1B"); //use database TargetLoc1B

            System.IO.StreamWriter resultFile = new System.IO.StreamWriter(@".\results.txt");
            System.IO.StreamReader queryFile = new System.IO.StreamReader(@".\benchmark_queries.sql");
            string queryString;
            while ((queryString = queryFile.ReadLine()) != null)
            {
                //execute query:
                client.Query(queryString);
                (Dictionary<string, List<object>> queryResult, Dictionary<string, float> executionTimes) result = (null, null);
                result = client.GetNextQueryResult().queryResult;

                //save query result to a file:
                resultFile.WriteLine(result.executionTimes.Values.Sum());
            }
            queryFile.Close();
            resultFile.Close();
        }    
    }
}
