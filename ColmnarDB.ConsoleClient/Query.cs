using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.Data;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using ColmnarDB.NetworkClient;
using ColmnarDB.NetworkClient.Message;
using Google.Protobuf.Collections;

namespace ColmnarDB.ConsoleClient
{
    /// <summary>
    /// Class to run entered query
    /// </summary>
    public class Query
    {
        public static readonly short numberOfQueryExec = 200;

        /// <summary>
        /// Run a query N times and print the query's average execution time.
        /// </summary>
        /// <param name="query">Executed query</param>
        public void RunTestQuery(string queryString, int consoleWidth, ColumnarDBClient client)
        {
            try
            {
                ColumnarDataTable result = null;
                Dictionary<string, float> executionTimes = null;
                float resultSum = 0;

                client.Query(queryString);

                while (((result, executionTimes) = client.GetNextQueryResult()).result != null)
                {
                    resultSum += executionTimes.Values.Sum();
                }

                Console.Out.WriteLine((resultSum).ToString() + " (first run)");
                resultSum = 0;

                //execute query N times (used cache):
                for (int i = 0; i < numberOfQueryExec; i++)
                {
                    client.Query(queryString);

                    while (((result, executionTimes) = client.GetNextQueryResult()).result != null)
                    {
                        resultSum += executionTimes.Values.Sum();
                    }                   
                }
                
                Console.WriteLine(SuccessfulQuery(queryString));
                Console.Out.WriteLine((resultSum / numberOfQueryExec) + " (average cached " + numberOfQueryExec + " runs)");
            }
            catch (IOException e)
            {
                Console.WriteLine(e.Message);
            }
            catch (QueryException e)
            {
                Console.WriteLine("Query Exception: " + e.Message);
            }
            catch (Exception e)
            {
                Console.WriteLine(UnknownException() + e.Message);
            }
        }

        /// <summary>
        /// Run a query one time and print result.
        /// </summary>
        /// <param name="query">Executed query</param>
        public void RunQuery(string query, int consoleWidth, ColumnarDBClient client)
        {
            try
            {
                ColumnarDataTable result = null;
                Dictionary<string, float> executionTimes = null;
                try
                {
                    client.Query(query);
                }
                catch (QueryException e)
                {
                    Console.WriteLine("Query Exception occured: " + e.Message);
                }
                Console.WriteLine(SuccessfulQuery(query));
                while (((result, executionTimes) = client.GetNextQueryResult()).result != null)
                {
                    PrintResults(result.GetColumnData(), result.GetOrderedColumnNames(), executionTimes, consoleWidth);
                }
            }
            catch (IOException e)
            {
                Console.WriteLine(e.Message);
            }
            catch (QueryException e)
            {
                Console.WriteLine("Query Exception: " + e.Message);
            }
            catch (Exception e)
            {
                Console.WriteLine(UnknownException() + e.Message);
            }
        }

        /// <summary>
        /// Returns output for successful query running
        /// </summary>
        /// <param name="query">Executed query</param>
        /// <returns>string about successful query running prepared for console output</returns>
        public static string SuccessfulQuery(string query)
        {
            return string.Format("Successful running entered query: '{0}'", query);
        }

        /// <summary>
        /// Return output when there is an unknown exception
        /// </summary>
        /// <returns>string about unknown exception prepared for console output</returns>
        public static string UnknownException()
        {
            return "Unknown exception has occured. Check if server is still running.";
        }

        /// <summary>
        /// Prints result in form of a table in console
        /// </summary>
        /// <param name="result">result of SQL query</param>
        /// <param name="time">time of executing query</param>
        /// <param name="consoleWidth">number of chars in row in console</param>
        public static void PrintResults(Dictionary<string, System.Collections.IList> result, List<string> orderedColumnNames, Dictionary<string, float> executionTimes, int consoleWidth)
        {
            var leftAlign = 3;
            var rightAlign = 20;
            string format = "{0,-" + leftAlign + "} {1,-" + rightAlign + "}";

            var numberOfRows = 0;
            foreach (var column in result.Keys)
            {
                numberOfRows = result[column].Count;
            }

            Console.WriteLine("Number of result rows: " + numberOfRows);
            Console.WriteLine("Time of query execution: ");

            foreach (var part in executionTimes)
            {
                Console.WriteLine(format, "", part.Key + ": " + part.Value + " ms");
            }
            Console.WriteLine(format, "", "Total execution time: " + executionTimes.Values.Sum() + " ms");

            if (numberOfRows == 0)
            {
                return;
            }

            var numberOfColumnsThatFitIntoConsole = (consoleWidth - 1) / (leftAlign + rightAlign + 1);

            if(numberOfColumnsThatFitIntoConsole == 0)
            {
                Console.Out.WriteLine("Your console window is not wide enough.");
            }

            else
            {
                //divide result, because there can be more columns that fit into console 
                while (orderedColumnNames.Count > numberOfColumnsThatFitIntoConsole)
                {
                    var orderedColumnThatFitIntoConsole = new List<string>();
                    Dictionary<string, System.Collections.IList> temp = new Dictionary<string, System.Collections.IList>();

                    for (int i = 0; i < numberOfColumnsThatFitIntoConsole; i++)
                    {
                        temp.Add(orderedColumnNames[0], result[orderedColumnNames[0]]);
                        orderedColumnThatFitIntoConsole.Add(orderedColumnNames[0]);
                        orderedColumnNames.RemoveAt(0);
                    }
                    PrintDividedOutput(temp, orderedColumnThatFitIntoConsole, numberOfRows, format, leftAlign, rightAlign);
                }
                PrintDividedOutput(result, orderedColumnNames, numberOfRows, format, leftAlign, rightAlign);
            }
        }

        /// <summary>
        /// Function to print part of table that fits into width of console
        /// </summary>
        /// <param name="result">Part of result to print</param>
        /// <param name="numberOfRows">Number of rows of result</param>
        /// <param name="format">format for output</param>
        /// <param name="leftAlign">left align of format</param>
        /// <param name="rightAlign">right align of format</param>
        public static void PrintDividedOutput(Dictionary<string, System.Collections.IList> result, List<string> orderedColumnNames, int numberOfRows, string format, int leftAlign, int rightAlign)
        {

            for (int i = 0; i < orderedColumnNames.Count; i++)
            {
                Console.Write("+");
                for (int j = 0; j < leftAlign + rightAlign; j++)
                {
                    Console.Write("-");
                }
            }

            Console.WriteLine("+");

            //Prints names of columns
            foreach (var column in orderedColumnNames)
            {
                string replacement = Regex.Replace(column, @"\t|\n|\r", "");
                if (replacement.Length > rightAlign)
                {
                    var newValue = replacement.Substring(0, rightAlign - 3) + "...";
                    Console.Write(format, "|", newValue);
                }
                else
                {
                    Console.Write(format, "|", replacement);
                }
            }

            Console.WriteLine("|");

            //Print line between names of columns and values
            //23 is a result of format sum 
            for (int i = 0; i < orderedColumnNames.Count; i++)
            {
                Console.Write("+");
                for (int j = 0; j < leftAlign + rightAlign; j++)
                {
                    Console.Write("-");
                }
            }

            Console.WriteLine("+");

            //Prints values of each column
            var columnIndex = 0;
            for (int i = 0; i < numberOfRows; i++)
            {
                foreach (var column in orderedColumnNames)
                {
                    columnIndex += 1;

                    string replacement = Regex.Replace((result[column][i] ?? "NULL").ToString(), @"\t|\n|\r", "");
                    if (replacement.Length > rightAlign)
                    {
                        var newValue = replacement.Substring(0, rightAlign - 3) + "...";
                        Console.Write(format, "|", newValue);
                    }
                    else
                    {
                        Console.Write(format, "|", replacement);
                    }
                }

                Console.WriteLine("|");
            }

            for (int i = 0; i < orderedColumnNames.Count; i++)
            {
                Console.Write("+");
                for (int j = 0; j < leftAlign + rightAlign; j++)
                {
                    Console.Write("-");
                }
            }

            Console.WriteLine("+");
        }
    }
}