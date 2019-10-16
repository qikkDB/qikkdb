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
        public static readonly string ipAddress = "127.0.0.1";
        public static readonly short port = 12345;

        public static readonly string resultFilePath = "results.txt";
        public static readonly int numberOfQueryExec = 200;

        public static readonly string telcoDbName = "TargetLocator";
        public static readonly string telcoQueriesPath = "../../../ColmnarDB.BenchmarkUtility/telco_queries.sql";

        public static readonly string geoDbName = "GeoTest";
        public static readonly string geoQueriesPath = "../../../ColmnarDB.BenchmarkUtility/geo_queries.sql";

        public static readonly string taxiDbName = "TaxiRides";
        public static readonly string taxiQueriesPath = "../../../ColmnarDB.BenchmarkUtility/taxi_queries.sql";

        public static readonly string stcsDbName = "stcs";
        public static readonly string stcsQueriesPath = "../../../ColmnarDB.BenchmarkUtility/stcs_queries.sql";

        /// <summary>
        /// Load benchmark queries from a file, execute them one by one and save results.
        /// </summary>
        public static void Main(string[] args)
        {
            Array.Sort(args);

            ColumnarDBClient client = new ColumnarDBClient("Host=" + ipAddress + ";" + "Port=" + port.ToString() + ";");
            client.Connect();
            Console.Out.WriteLine("Client has successfully connected to server.");

            UseDatabase use = new UseDatabase();

            if (File.Exists(resultFilePath))
            {
                File.Delete(resultFilePath);
            }

            var resultFile = new System.IO.StreamWriter(resultFilePath);

            if (args.GetLength(0) == 0)
            {
                string[] array = {"-a"};
                List<string> tempList = new List<string>(array);
                args = tempList.ToArray();
            }

            resultFile.WriteLine("Each query was executed " + numberOfQueryExec + " times and the average time was saved.");
            
            foreach (string arg in args)
            {
                string queryString;

                if (String.Equals(arg, "-a") || String.Equals(arg, "-b"))
                {
                    //test telco queries:
                    use.Use(telcoDbName, client);
  
                    client.UseDatabase(telcoDbName);

                    var queryFile = new System.IO.StreamReader(telcoQueriesPath);
                    Console.Out.WriteLine("Benchmark queries from file '" + telcoQueriesPath + "' were loaded.");

                    while ((queryString = queryFile.ReadLine()) != null)
                    {
                        Console.Out.WriteLine("Executing benchmark query: " + queryString);
                        resultFile.WriteLine(queryString);

                        //execute query first time (no cache):
                        float resultSum = 0;
                        client.Query(queryString);
                        Dictionary<string, float> executionTimes = null;
                        ColumnarDataTable result = null;

                        while (((result, executionTimes) = client.GetNextQueryResult()).result != null)
                        {
                            resultSum += executionTimes.Values.Sum();
                        }
                
                        //save query result to a file:
                        resultFile.WriteLine((resultSum).ToString() + " (first run)");
                        Console.Out.WriteLine((resultSum).ToString() + " (first run)"); 

                        //execute query N times (used cache):
                        for (int i = 0; i < numberOfQueryExec; i++)
                        {
                            client.Query(queryString);

                            while (((result, executionTimes) = client.GetNextQueryResult()).result != null)
                            {
                                resultSum += executionTimes.Values.Sum();
                            }                   
                        }

                        double avgQueryExec = resultSum / numberOfQueryExec;

                        //save query result to a file:
                        resultFile.WriteLine(avgQueryExec.ToString() + " (average cached N runs)");
                        Console.Out.WriteLine(avgQueryExec + " (average cached N runs)");

                        //check if query execution time is acceptable and save the result:
                        int queryExpectedExecTime = System.Convert.ToInt32(queryFile.ReadLine());
                        if (avgQueryExec < queryExpectedExecTime)
                        {
                            resultFile.WriteLine("The query '" + queryString + "' has passed the execution time test. Expected / Actual average query execution time: " + queryExpectedExecTime.ToString() + " / " + avgQueryExec.ToString());
                            Console.Out.WriteLine("The query '" + queryString + "' has passed the execution time test. Expected / Actual average query execution time: " + queryExpectedExecTime + " / " + avgQueryExec);
                        }
                        else
                        {
                            resultFile.WriteLine("The query '" + queryString + "' has FAILED the execution time test. Expected / Actual average query execution time: " + queryExpectedExecTime.ToString() + " / " + avgQueryExec.ToString());
                            Console.Out.WriteLine("The query '" + queryString + "' has FAILED the execution time test. Expected / Actual average query execution time: " + queryExpectedExecTime + " / " + avgQueryExec);
                            Environment.Exit(1);
                        }
                    }
                    queryFile.Close();
                }

                if (String.Equals(arg, "-a") || String.Equals(arg, "-g"))
                {
                    //test geo queries:
                    use.Use(geoDbName, client);
            
                    client.UseDatabase(geoDbName);

                    var queryFile = new System.IO.StreamReader(geoQueriesPath);
                    Console.Out.WriteLine("Benchmark queries from file '" + geoQueriesPath + "' were loaded.");

                    while ((queryString = queryFile.ReadLine()) != null)
                    {
                        Console.Out.WriteLine("Executing benchmark query: " + queryString);
                        resultFile.WriteLine(queryString);

                        //execute query first time (no cache):
                        float resultSum = 0;
                        client.Query(queryString);
                        Dictionary<string, float> executionTimes = null;
                        ColumnarDataTable result = null;

                        while (((result, executionTimes) = client.GetNextQueryResult()).result != null)
                        {
                            resultSum += executionTimes.Values.Sum();
                        }
                
                        //save query result to a file:
                        resultFile.WriteLine((resultSum).ToString() + " (first run)");
                        Console.Out.WriteLine((resultSum).ToString() + " (first run)"); 

                        //execute query N times (used cache):
                        for (int i = 0; i < numberOfQueryExec; i++)
                        {
                            client.Query(queryString);

                            while (((result, executionTimes) = client.GetNextQueryResult()).result != null)
                            {
                                resultSum += executionTimes.Values.Sum();
                            }                   
                        }

                        double avgQueryExec = resultSum / numberOfQueryExec;

                        //save query result to a file:
                        resultFile.WriteLine(avgQueryExec.ToString() + " (average cached N runs)");
                        Console.Out.WriteLine(avgQueryExec + " (average cached N runs)");

                        //check if query execution time is acceptable and save the result:
                        int queryExpectedExecTime = System.Convert.ToInt32(queryFile.ReadLine());
                        if (avgQueryExec < queryExpectedExecTime)
                        {
                            resultFile.WriteLine("The query '" + queryString + "' has passed the execution time test. Expected / Actual average query execution time: " + queryExpectedExecTime.ToString() + " / " + avgQueryExec.ToString());
                            Console.Out.WriteLine("The query '" + queryString + "' has passed the execution time test. Expected / Actual average query execution time: " + queryExpectedExecTime + " / " + avgQueryExec);
                        }
                        else
                        {
                            resultFile.WriteLine("The query '" + queryString + "' has FAILED the execution time test. Expected / Actual average query execution time: " + queryExpectedExecTime.ToString() + " / " + avgQueryExec.ToString());
                            Console.Out.WriteLine("The query '" + queryString + "' has FAILED the execution time test. Expected / Actual average query execution time: " + queryExpectedExecTime + " / " + avgQueryExec);
                            Environment.Exit(1);
                        }
                    }
                    queryFile.Close();
                }

                if (String.Equals(arg, "-a") || String.Equals(arg, "-t"))
                {
                    //test taxi queries:
                    use.Use(taxiDbName, client);
            
                    client.UseDatabase(taxiDbName);

                    var queryFile = new System.IO.StreamReader(taxiQueriesPath);
                    Console.Out.WriteLine("Benchmark queries from file '" + taxiQueriesPath + "' were loaded.");

                    while ((queryString = queryFile.ReadLine()) != null)
                    {
                        Console.Out.WriteLine("Executing benchmark query: " + queryString);
                        resultFile.WriteLine(queryString);

                        //execute query first time (no cache):
                        float resultSum = 0;
                        client.Query(queryString);
                        Dictionary<string, float> executionTimes = null;
                        ColumnarDataTable result = null;

                        while (((result, executionTimes) = client.GetNextQueryResult()).result != null)
                        {
                            resultSum += executionTimes.Values.Sum();
                        }
                
                        //save query result to a file:
                        resultFile.WriteLine((resultSum).ToString() + " (first run)");
                        Console.Out.WriteLine((resultSum).ToString() + " (first run)"); 

                        //execute query N times (used cache):
                        for (int i = 0; i < numberOfQueryExec; i++)
                        {
                            client.Query(queryString);

                            while (((result, executionTimes) = client.GetNextQueryResult()).result != null)
                            {
                                resultSum += executionTimes.Values.Sum();
                            }                   
                        }

                        double avgQueryExec = resultSum / numberOfQueryExec;

                        //save query result to a file:
                        resultFile.WriteLine(avgQueryExec.ToString() + " (average cached N runs)");
                        Console.Out.WriteLine(avgQueryExec + " (average cached N runs)");

                        //check if query execution time is acceptable and save the result:
                        int queryExpectedExecTime = System.Convert.ToInt32(queryFile.ReadLine());
                        if (avgQueryExec < queryExpectedExecTime)
                        {
                            resultFile.WriteLine("The query '" + queryString + "' has passed the execution time test. Expected / Actual average query execution time: " + queryExpectedExecTime.ToString() + " / " + avgQueryExec.ToString());
                            Console.Out.WriteLine("The query '" + queryString + "' has passed the execution time test. Expected / Actual average query execution time: " + queryExpectedExecTime + " / " + avgQueryExec);
                        }
                        else
                        {
                            resultFile.WriteLine("The query '" + queryString + "' has FAILED the execution time test. Expected / Actual average query execution time: " + queryExpectedExecTime.ToString() + " / " + avgQueryExec.ToString());
                            Console.Out.WriteLine("The query '" + queryString + "' has FAILED the execution time test. Expected / Actual average query execution time: " + queryExpectedExecTime + " / " + avgQueryExec);
                            Environment.Exit(1);
                        }
                    }
                    queryFile.Close();
                }

                if (String.Equals(arg, "-a") || String.Equals(arg, "-s"))
                {
                    //test stcs queries:
                    use.Use(stcsDbName, client);
            
                    client.UseDatabase(stcsDbName);

                    var queryFile = new System.IO.StreamReader(stcsQueriesPath);
                    Console.Out.WriteLine("Benchmark queries from file '" + stcsQueriesPath + "' were loaded.");

                    while ((queryString = queryFile.ReadLine()) != null)
                    {
                        Console.Out.WriteLine("Executing benchmark query: " + queryString);
                        resultFile.WriteLine(queryString);

                        //execute query first time (no cache):
                        float resultSum = 0;
                        client.Query(queryString);
                        Dictionary<string, float> executionTimes = null;
                        ColumnarDataTable result = null;

                        while (((result, executionTimes) = client.GetNextQueryResult()).result != null)
                        {
                            resultSum += executionTimes.Values.Sum();
                        }
                
                        //save query result to a file:
                        resultFile.WriteLine((resultSum).ToString() + " (first run)");
                        Console.Out.WriteLine((resultSum).ToString() + " (first run)"); 

                        //execute query N times (used cache):
                        for (int i = 0; i < numberOfQueryExec; i++)
                        {
                            client.Query(queryString);

                            while (((result, executionTimes) = client.GetNextQueryResult()).result != null)
                            {
                                resultSum += executionTimes.Values.Sum();
                            }                   
                        }

                        double avgQueryExec = resultSum / numberOfQueryExec;

                        //save query result to a file:
                        resultFile.WriteLine(avgQueryExec.ToString() + " (average cached N runs)");
                        Console.Out.WriteLine(avgQueryExec + " (average cached N runs)");

                        //check if query execution time is acceptable and save the result:
                        int queryExpectedExecTime = System.Convert.ToInt32(queryFile.ReadLine());
                        if (avgQueryExec < queryExpectedExecTime)
                        {
                            resultFile.WriteLine("The query '" + queryString + "' has passed the execution time test. Expected / Actual average query execution time: " + queryExpectedExecTime.ToString() + " / " + avgQueryExec.ToString());
                            Console.Out.WriteLine("The query '" + queryString + "' has passed the execution time test. Expected / Actual average query execution time: " + queryExpectedExecTime + " / " + avgQueryExec);
                        }
                        else
                        {
                            resultFile.WriteLine("The query '" + queryString + "' has FAILED the execution time test. Expected / Actual average query execution time: " + queryExpectedExecTime.ToString() + " / " + avgQueryExec.ToString());
                            Console.Out.WriteLine("The query '" + queryString + "' has FAILED the execution time test. Expected / Actual average query execution time: " + queryExpectedExecTime + " / " + avgQueryExec);
                            Environment.Exit(1);
                        }
                    }
                    queryFile.Close();
                }
            }
            resultFile.Close();
        }    
    }
}
