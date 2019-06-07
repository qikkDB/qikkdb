using System;
using System.Collections.Generic;
using ColmnarDB.NetworkClient;
using System.Linq;

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

            //TODO toto zmenit na prijatie argumentu z mainu !!!!!!!!
            client.ImportCSV("TargetLoc1M", "test-data/TargetLoc1M.csv");

            client.UseDatabase("TargetLoc1M");

            System.IO.StreamWriter resultFile = new System.IO.StreamWriter("ColmnarDB.BenchmarkUtility/results.txt");
            System.IO.StreamReader queryFile = new System.IO.StreamReader("ColmnarDB.BenchmarkUtility/benchmark_queries.sql");
            string queryString;
            while ((queryString = queryFile.ReadLine()) != null)
            {
                //execute query:
                client.Query(queryString);
                (Dictionary<string, List<object>> queryResult, Dictionary<string, float> executionTimes) result = (null, null);
                result = client.GetNextQueryResult();

                //save query result to a file:
                resultFile.WriteLine(result.executionTimes.Values.Sum());
            }
            queryFile.Close();
            resultFile.Close();
        }    
    }
}
