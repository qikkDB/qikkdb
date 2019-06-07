using System;
using System.Collections.Generic;
using ColmnarDB.NetworkClient;
using System.Linq;
using System.IO;

namespace ColmnarDB.BenchmarkUtility
{
    public class Program
    {
        public static readonly string IpAddress = "127.0.0.1";
        public static readonly short Port = 12345;

        public static readonly string resultFilePath = "results.txt";
        public static readonly string csvFilePath = "test-data/TargetLoc1B.csv";
        public static readonly string databaseName = "TargetLoc1B";
        public static readonly string benchmarkFilePath = "benchmark_queries.sql";

        /// <summary>
        /// Load benchmark queries from a file, execute them one by one and save results.
        /// </summary>
        public static void Main(string[] args)
        {
            ColumnarDBClient client = new ColumnarDBClient(IpAddress, Port);
            client.Connect();
            Console.Out.WriteLine("Client has successfully connected to server.");

            if (File.Exists(resultFilePath))
            {
                File.Delete(resultFilePath);
            }

            var resultFile = new System.IO.StreamWriter(resultFilePath);

            //TODO toto zmenit na prijatie argumentu z mainu !!!!!!!!
            Console.Out.WriteLine("Starting importing CSV file named: " + csvFilePath);
            var start = DateTime.Now;
            client.ImportCSV(databaseName, csvFilePath);
            var timeDiff = DateTime.Now - start;
            Console.Out.WriteLine("Importing of CSV file has successfully finished.");

            resultFile.WriteLine("Importing CSV file (" + csvFilePath + ") took: " + timeDiff.ToString() + " ms.");

            client.UseDatabase(databaseName);

            Console.Out.WriteLine("Database was successfully loaded.");

            var queryFile = new System.IO.StreamReader(benchmarkFilePath);
            Console.Out.WriteLine("Benchmark queries from file '" + benchmarkFilePath + "' were loaded.");

            string queryString;
            while ((queryString = queryFile.ReadLine()) != null)
            {
                Console.Out.WriteLine("Executing benchmark query: " + queryString);

                //execute query:
                client.Query(queryString);
                (Dictionary<string, List<object>> queryResult, Dictionary<string, float> executionTimes) result = (null, null);
                result = client.GetNextQueryResult();

                //save query result to a file:
                resultFile.WriteLine(result.executionTimes.Values.Sum().ToString());
            }
            queryFile.Close();
            resultFile.Close();
        }    
    }
}
