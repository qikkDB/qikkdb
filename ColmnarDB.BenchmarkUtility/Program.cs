using System;
using System.Collections.Generic;
using ColmnarDB.NetworkClient;
using ColmnarDB.ConsoleClient;
using System.Linq;
using System.IO;
using System.Collections;
using System.Globalization;

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
        public static int Main(string[] args)
        {
            bool avgTimePassed = true;
            bool correctResultsPassed = true;

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
                string queryExptectedString;

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
                        try
                        {
                            client.Query(queryString);
                        }
                        catch (Exception e)
                        {
                            Console.Out.WriteLine(e);
                        }
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
                            try
                            {
                                client.Query(queryString);
                            }
                            catch (Exception e)
                            {
                                Console.Out.WriteLine(e);
                            }

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
                            avgTimePassed = false;
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
                        try
                        {
                            client.Query(queryString);
                        }
                        catch (Exception e)
                        {
                            Console.Out.WriteLine(e);
                        }
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
                            try
                            {
                                client.Query(queryString);
                            }
                            catch (Exception e)
                            {
                                Console.Out.WriteLine(e);
                            }

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
                            avgTimePassed = false;
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

                    int queryIndex = 1; //indexing from 1, not zero, because we use to call it taxi rides query number 1, not number 0, so it is not confusing

                    while ((queryString = queryFile.ReadLine()) != null)
                    {
                        Console.Out.WriteLine("Executing benchmark query: " + queryString);
                        resultFile.WriteLine(queryString);

                        //execute query first time (no cache):
                        float resultSum = 0;
                        try
                        {
                            client.Query(queryString);
                        }
                        catch (Exception e)
                        {
                            Console.Out.WriteLine(e);
                        }
                        Dictionary<string, float> executionTimes = null;
                        ColumnarDataTable result = null;
                        ColumnarDataTable oldResult = null;

                        while (((result, executionTimes) = client.GetNextQueryResult()).result != null)
                        {
                            oldResult = result;
                            resultSum += executionTimes.Values.Sum();
                        }

                        //read file where the results of a particular query are saved:
                        var queryExpectedResultFile = new System.IO.StreamReader("../../../ColmnarDB.BenchmarkUtility/" + taxiDbName + "_testQuery_" + queryIndex + ".txt");
                        queryIndex++;

                        //read the file header
                        var expectedColumnNames = queryExpectedResultFile.ReadLine().Split('|');

                        //read expected column data types
                        var expectedDataTypes= queryExpectedResultFile.ReadLine().Split('|');

                        Dictionary<string, IList> exptectedColumns = new Dictionary<string, IList>();

                        for (int i = 0; i < expectedColumnNames.Length; i++)
                        {
                            switch (expectedDataTypes[i])
                            {
                                case "INT":
                                    exptectedColumns.Add(expectedColumnNames[i], new List<Int32>());
                                    break;
                                case "LONG":
                                    exptectedColumns.Add(expectedColumnNames[i], new List<Int64>());
                                    break;
                                case "FLOAT":
                                    exptectedColumns.Add(expectedColumnNames[i], new List<Single>());
                                    break;
                                case "DOUBLE":
                                    exptectedColumns.Add(expectedColumnNames[i], new List<Double>());
                                    break;
                                case "STRING":
                                    exptectedColumns.Add(expectedColumnNames[i], new List<String>());
                                    break;
                            }
                            
                        }

                        //read results from a file:
                        while ((queryExptectedString = queryExpectedResultFile.ReadLine()) != null)
                        {
                            var results = queryExptectedString.Split('|');

                            for (int i = 0; i < expectedColumnNames.Length; i++)
                            {
                                switch (expectedDataTypes[i])
                                {
                                    case "INT":
                                        exptectedColumns[expectedColumnNames[i]].Add(Int32.Parse(results[i]));
                                        break;
                                    case "LONG":
                                        exptectedColumns[expectedColumnNames[i]].Add(Int64.Parse(results[i]));
                                        break;
                                    case "FLOAT":
                                        exptectedColumns[expectedColumnNames[i]].Add(Single.Parse(results[i]));
                                        break;
                                    case "DOUBLE":
                                        var styles = NumberStyles.AllowDecimalPoint;
                                        var provider = CultureInfo.CreateSpecificCulture("en-US");
                                        exptectedColumns[expectedColumnNames[i]].Add(Double.Parse(results[i], styles, provider));
                                        break;
                                    case "STRING":
                                        exptectedColumns[expectedColumnNames[i]].Add(results[i]);
                                        break;
                                }
                            }
                        }

                        //check if the expected result dictionary is the same as actual query result dictionary:
                        for (int i = 0; i < expectedColumnNames.Length; i++)
                        {
                            try
                            {
                                if (exptectedColumns[expectedColumnNames[i]].Count != oldResult.GetColumnData()[expectedColumnNames[i]].Count)
                                {
                                    resultFile.WriteLine("The query '" + queryString + "' has FAILED the correct results test. Expected / Actual count of result entries: " + exptectedColumns[expectedColumnNames[i]].Count.ToString() + " / " + oldResult.GetColumnData()[expectedColumnNames[i]].Count.ToString());
                                    Console.Out.WriteLine("The query '" + queryString + "' has FAILED the correct results test. Expected / Actual count of result entries: " + exptectedColumns[expectedColumnNames[i]].Count + " / " + oldResult.GetColumnData()[expectedColumnNames[i]].Count);
                                    correctResultsPassed = false;
                                }
                                else
                                {
                                    bool tempCorrectResultsPassed = true;

                                    //check each element in result's lists
                                    for (int j = 0; j < exptectedColumns[expectedColumnNames[i]].Count; j++)
                                    {
                                        switch (expectedDataTypes[i])
                                        {
                                            case "INT":
                                                if ((int)exptectedColumns[expectedColumnNames[i]][j] != (int)oldResult.GetColumnData()[expectedColumnNames[i]][j])
                                                {
                                                    tempCorrectResultsPassed = false;
                                                }
                                                break;
                                            case "LONG":
                                                if ((long)exptectedColumns[expectedColumnNames[i]][j] != (long)oldResult.GetColumnData()[expectedColumnNames[i]][j])
                                                {
                                                    tempCorrectResultsPassed = false;
                                                }
                                                break;
                                            case "FLOAT":
                                                if (Math.Abs((float)exptectedColumns[expectedColumnNames[i]][j] - (float)oldResult.GetColumnData()[expectedColumnNames[i]][j]) > 0.001)
                                                {
                                                    tempCorrectResultsPassed = false;
                                                }
                                                break;
                                            case "DOUBLE":
                                                if (Math.Abs((double)exptectedColumns[expectedColumnNames[i]][j] - (double)oldResult.GetColumnData()[expectedColumnNames[i]][j]) > 0.001)
                                                {
                                                    tempCorrectResultsPassed = false;
                                                }
                                                break;
                                            case "STRING":
                                                if ((string)exptectedColumns[expectedColumnNames[i]][j] != (string)oldResult.GetColumnData()[expectedColumnNames[i]][j])
                                                {
                                                    tempCorrectResultsPassed = false;
                                                }
                                                break;
                                        }

                                        if (!tempCorrectResultsPassed)
                                        {
                                            resultFile.WriteLine("The query '" + queryString + "' has FAILED the correct results test. Expected / Actual returned value: " + exptectedColumns[expectedColumnNames[i]][j].ToString() + " / " + oldResult.GetColumnData()[expectedColumnNames[i]][j].ToString());
                                            Console.Out.WriteLine("The query '" + queryString + "' has FAILED the correct results test. Expected / Actual returned value: " + exptectedColumns[expectedColumnNames[i]][j] + " / " + oldResult.GetColumnData()[expectedColumnNames[i]][j]);
                                            correctResultsPassed = false;
                                        }
                                    }
                                }
                            }
                            catch (System.Collections.Generic.KeyNotFoundException)
                            {
                                resultFile.WriteLine("The query '" + queryString + "' has FAILED the correct results test. Expected / Actual returned value: " + exptectedColumns[expectedColumnNames[i]].ToString() + " / System.Collections.Generic.KeyNotFoundException was thrown.");
                                Console.Out.WriteLine("The query '" + queryString + "' has FAILED the correct results test. Expected / Actual returned value: " + exptectedColumns[expectedColumnNames[i]] + " / System.Collections.Generic.KeyNotFoundException was thrown.");
                                correctResultsPassed = false;
                            }
                        }
                
                        //save query result to a file:
                        resultFile.WriteLine((resultSum).ToString() + " (first run)");
                        Console.Out.WriteLine((resultSum).ToString() + " (first run)"); 

                        //execute query N times (used cache):
                        for (int i = 0; i < numberOfQueryExec; i++)
                        {
                            try
                            {
                                client.Query(queryString);
                            }
                            catch (Exception e)
                            {
                                Console.Out.WriteLine(e);
                            }

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
                            avgTimePassed = false;
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
                        try
                        {
                            client.Query(queryString);
                        }
                        catch (Exception e)
                        {
                            Console.Out.WriteLine(e);
                        }
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
                            try
                            {
                                client.Query(queryString);
                            }
                            catch (Exception e)
                            {
                                Console.Out.WriteLine(e);
                            }

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
                            avgTimePassed = false;
                        }
                    }
                    queryFile.Close();
                }
            }
            resultFile.Close();

            //return exit code:
            if (correctResultsPassed && avgTimePassed)
            {
                //everything was successful
                return 0;
            }

            if (!correctResultsPassed && avgTimePassed)
            {
                //query results were not correct, but query has finished in expected time
                return 1;
            }

            if (correctResultsPassed && !avgTimePassed)
            {
                //query results were corrcet, but query has not finished in expected time
                return 2;
            }

            if (!correctResultsPassed && !avgTimePassed)
            {
                //neither query results were correct, nor query has finished execution in expected time
                return 3;
            }

            //something else has happend
            return 4;
        }    
    }
}
