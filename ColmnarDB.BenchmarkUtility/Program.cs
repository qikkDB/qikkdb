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
        public static readonly int numberOfQueryExec = 5;

        public static readonly string telcoDataCsvPath = "test-data/TargetLoc1B.csv";
        public static readonly string telcoDbName = "TargetLoc1B";
        public static readonly string telcoQueriesPath = "ColmnarDB.BenchmarkUtility/telco_queries.sql";

        public static readonly string geoDataCsvPath = "test-data/zones335.csv";
        public static readonly string geoDbName = "zones335";
        public static readonly string geoQueriesPath = "ColmnarDB.BenchmarkUtility/geo_queries.sql";

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

            //test telco queries:
            Console.Out.WriteLine("Importing CSV file named: " + telcoDataCsvPath + ". Please wait.");
            var start = DateTime.Now;
            client.ImportCSV(telcoDbName, telcoDataCsvPath);
            var timeDiff = DateTime.Now - start;
            Console.Out.WriteLine("Importing of CSV file has successfully finished.");

            resultFile.WriteLine("Each query was executed " + numberOfQueryExec + " times and the average time was saved.");
            resultFile.WriteLine("Importing CSV file (" + telcoDataCsvPath + ") took: " + timeDiff.ToString() + " (HH:MM:SS).");

            client.UseDatabase(telcoDbName);

            Console.Out.WriteLine("Database was successfully loaded.");

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

                foreach (var key in result.executionTimes.Keys)
                {
                    Console.Out.WriteLine("Execution times key: " + key);
                }
                
                
                //save query result to a file:
                resultFile.WriteLine((resultSum).ToString() + " (first run)");

                resultSum = 0;

                //execute query n times (used cache):
                for (int i = 0; i < numberOfQueryExec; i++)
                {
                    client.Query(queryString);
                    result = client.GetNextQueryResult();
                    resultSum += result.executionTimes.Values.Sum();
                } 

                //save query result to a file:
                resultFile.WriteLine((resultSum / numberOfQueryExec).ToString() + " (average cached run)");
            }
            queryFile.Close();

            //test geo queries:
            Console.Out.WriteLine("Starting importing CSV file named: " + geoDataCsvPath);
            start = DateTime.Now;
            client.ImportCSV(geoDbName, geoDataCsvPath);
            timeDiff = DateTime.Now - start;
            Console.Out.WriteLine("Importing of CSV file has successfully finished.");

            resultFile.WriteLine("Importing CSV file (" + geoDataCsvPath + ") took: " + timeDiff.ToString() + " (HH:MM:SS).");

            client.UseDatabase(geoDbName);

            Console.Out.WriteLine("Database was successfully loaded.");

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

                resultSum = 0;

                //execute query n times (used cache):
                for (int i = 0; i < numberOfQueryExec; i++)
                {
                    client.Query(queryString);
                    result = client.GetNextQueryResult();
                    resultSum += result.executionTimes.Values.Sum();
                }

                //save query result to a file:
                resultFile.WriteLine((resultSum / numberOfQueryExec).ToString() + " (average cached run)");
            }
            queryFile.Close();
            resultFile.Close();

            //drop databases:
            queryString = "DROP DATABASE " + telcoDbName + ";";
            client.Query(queryString);

            queryString = "DROP DATABASE " + geoDbName + ";";
            client.Query(queryString);
        }    
    }
}
