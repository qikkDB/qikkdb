using System;
using System.Collections.Generic;
using ColmnarDB.NetworkClient;
using ColmnarDB.ConsoleClient;
using System.Linq;
using System.IO;

namespace ColmnarDB.BenchmarkUtility
{
    public class Program
    {
        public static readonly string IpAddress = "127.0.0.1";
        public static readonly short Port = 12345;

        public static readonly string resultFilePath = "results.txt";
        public static readonly int numberOfQueryExec = 5;

        public static readonly string telcoDbName = "TargetLocator";
        public static readonly string telcoQueriesPath = "ColmnarDB.BenchmarkUtility/telco_queries.sql";

        public static readonly string geoDbName = "GeoTest";
        public static readonly string geoQueriesPath = "ColmnarDB.BenchmarkUtility/geo_queries.sql";

        public static readonly string taxiDbName = "TaxiRides";
        public static readonly string taxiQueriesPath = "ColmnarDB.BenchmarkUtility/taxi_queries.sql";

        /// <summary>
        /// Load benchmark queries from a file, execute them one by one and save results.
        /// </summary>
        public static void Main(string[] args)
        {
            ColumnarDBClient client = new ColumnarDBClient(IpAddress, Port);
            client.Connect();
            Console.Out.WriteLine("Client has successfully connected to server.");
            client.Query("show databases;");

            UseDatabase use = new UseDatabase();

            if (File.Exists(resultFilePath))
            {
                File.Delete(resultFilePath);
            }

            var resultFile = new System.IO.StreamWriter(resultFilePath);

            //test telco queries:
            use.Use(telcoDbName, client);
            resultFile.WriteLine("Each query was executed " + numberOfQueryExec + " times and the average time was saved.");
            
            client.UseDatabase(telcoDbName);

            var queryFile = new System.IO.StreamReader(telcoQueriesPath);
            Console.Out.WriteLine("Benchmark queries from file '" + telcoQueriesPath + "' were loaded.");

            string queryString;

            while ((queryString = queryFile.ReadLine()) != null)
            {
                Console.Out.WriteLine("Executing benchmark query: " + queryString);
                resultFile.WriteLine(queryString);

                //execute query first time (no cache):
                client.Query(queryString);
                (Dictionary<string, List<object>> queryResult, Dictionary<string, float> executionTimes) result = (null, null);
                result = client.GetNextQueryResult();
                float resultSum = result.executionTimes.Values.Sum();
                
                //save query result to a file:
                resultFile.WriteLine((resultSum).ToString() + " (first run)");
                Console.Out.WriteLine((resultSum).ToString() + " (first run)");

                resultSum = 0;

                //execute query n times (used cache):
                for (int i = 0; i < numberOfQueryExec; i++)
                {
                    client.Query(queryString);
                    result = client.GetNextQueryResult();
                    resultSum += result.executionTimes.Values.Sum();
                } 

                //save query result to a file:
                resultFile.WriteLine((resultSum / numberOfQueryExec).ToString() + " (average cached N runs)");
                Console.Out.WriteLine((resultSum / numberOfQueryExec) + " (average cached N runs)");
            }
            queryFile.Close();

            //test geo queries:
            use.Use(geoDbName, client);
            
            client.UseDatabase(geoDbName);

            queryFile = new System.IO.StreamReader(geoQueriesPath);
            Console.Out.WriteLine("Benchmark queries from file '" + geoQueriesPath + "' were loaded.");

            while ((queryString = queryFile.ReadLine()) != null)
            {
                Console.Out.WriteLine("Executing benchmark query: " + queryString);
                resultFile.WriteLine(queryString);

                //execute query first time (no cache):
                client.Query(queryString);
                (Dictionary<string, List<object>> queryResult, Dictionary<string, float> executionTimes) result = (null, null);
                result = client.GetNextQueryResult();
                float resultSum = result.executionTimes.Values.Sum();

                //save query result to a file:
                resultFile.WriteLine((resultSum).ToString() + " (first run)");
                Console.Out.WriteLine((resultSum).ToString() + " (first run)");

                resultSum = 0;

                //execute query n times (used cache):
                for (int i = 0; i < numberOfQueryExec; i++)
                {
                    client.Query(queryString);
                    result = client.GetNextQueryResult();
                    resultSum += result.executionTimes.Values.Sum();
                }

                //save query result to a file:
                resultFile.WriteLine((resultSum / numberOfQueryExec).ToString() + " (average cached N runs)");
                Console.Out.WriteLine((resultSum / numberOfQueryExec) + " (average cached N runs)");
            }
            queryFile.Close();

            //test taxi queries:
            use.Use(taxiDbName, client);
            
            client.UseDatabase(taxiDbName);

            queryFile = new System.IO.StreamReader(taxiQueriesPath);
            Console.Out.WriteLine("Benchmark queries from file '" + taxiQueriesPath + "' were loaded.");

            while ((queryString = queryFile.ReadLine()) != null)
            {
                Console.Out.WriteLine("Executing benchmark query: " + queryString);
                resultFile.WriteLine(queryString);

                //execute query first time (no cache):
                client.Query(queryString);
                (Dictionary<string, List<object>> queryResult, Dictionary<string, float> executionTimes) result = (null, null);
                result = client.GetNextQueryResult();
                float resultSum = result.executionTimes.Values.Sum();

                //save query result to a file:
                resultFile.WriteLine((resultSum).ToString() + " (first run)");
                Console.Out.WriteLine((resultSum).ToString() + " (first run)");

                resultSum = 0;

                //execute query n times (used cache):
                for (int i = 0; i < numberOfQueryExec; i++)
                {
                    client.Query(queryString);
                    result = client.GetNextQueryResult();
                    resultSum += result.executionTimes.Values.Sum();
                }

                //save query result to a file:
                resultFile.WriteLine((resultSum / numberOfQueryExec).ToString() + " (average cached N runs)");
                Console.Out.WriteLine((resultSum / numberOfQueryExec) + " (average cached N runs)");
            }
            queryFile.Close();
            resultFile.Close();
        }    
    }
}
