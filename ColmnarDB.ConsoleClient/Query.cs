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
        public Dictionary<string, List<object>> RunQuery(string query, int consoleWidth, ColumnarDBClient client)
        {
            Dictionary<string, List<object>> queryResult = new Dictionary<string, List<object>>();
            try
            {
                (Dictionary<string, List<object>> queryResult, Dictionary<string, float> executionTimes) result = (null, null);
                client.Query(query);
                Console.WriteLine(SuccessfulQuery(query));
                while ((result = client.GetNextQueryResult()).queryResult != null)
                {
                    PrintResults(result.queryResult, result.executionTimes, consoleWidth);
                    foreach (var kv in result.queryResult)
                    {
                        if (queryResult.ContainsKey(kv.Key))
                        {
                            queryResult[kv.Key].AddRange(kv.Value);
                        }
                        else
                        {
                            queryResult.Add(kv.Key, kv.Value);
                        }
                    }
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

            return queryResult;
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
        public static void PrintResults(Dictionary<string, List<object>> result, Dictionary<string,float> executionTimes, int consoleWidth)
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
                Console.WriteLine(format,"",part.Key + ": " + part.Value + " ms");
            }
            Console.WriteLine(format,"","Total execution time: " + executionTimes.Values.Sum() + " ms");
            
            if (numberOfRows == 0)
            {
                return;
            }

            var numberOfColumnsThatFitIntoConsole =(consoleWidth-1) / (leftAlign+rightAlign+1);

            //divide result, because there can be more columns that fit into console 
            while (result.Keys.Count > numberOfColumnsThatFitIntoConsole)
            {
                Dictionary<string, List<object>> temp = result.Take(numberOfColumnsThatFitIntoConsole).ToDictionary(c => c.Key, c => c.Value);
                result = result.Skip(numberOfColumnsThatFitIntoConsole).ToDictionary(c => c.Key, c => c.Value);
                
                PrintDividedOutput(temp,numberOfRows,format,leftAlign,rightAlign);
            }
            PrintDividedOutput(result,numberOfRows,format,leftAlign,rightAlign);
        }

        /// <summary>
        /// Function to print part of table that fits into width of console
        /// </summary>
        /// <param name="result">Part of result to print</param>
        /// <param name="numberOfRows">Number of rows of result</param>
        /// <param name="format">format for output</param>
        /// <param name="leftAlign">left align of format</param>
        /// <param name="rightAlign">right align of format</param>
        public static void PrintDividedOutput(Dictionary<string, List<object>> result, int numberOfRows, string format, int leftAlign, int rightAlign)
        {
            for (int i = 0; i < result.Keys.Count; i++)
                {
                    Console.Write("+");
                    for (int j = 0; j < leftAlign+rightAlign; j++)
                    {
                        Console.Write("-");
                    }
                }

                Console.WriteLine("+");

                //Prints names of columns
                foreach (var column in result.Keys)
                {
                    string replacement = Regex.Replace(column, @"\t|\n|\r", "");
                    if (replacement.Length > rightAlign)
                    {
                        var newValue = replacement.Substring(0, rightAlign-3) + "...";
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
                for (int i = 0; i < result.Keys.Count; i++)
                {
                    Console.Write("+");
                    for (int j = 0; j < leftAlign+rightAlign; j++)
                    {
                        Console.Write("-");
                    }
                }

                Console.WriteLine("+");

                //Prints values of each column
                var columnIndex = 0;
                for (int i = 0; i < numberOfRows; i++)
                {
                    foreach (var column in result.Keys)
                    {
                        columnIndex += 1;
                        string replacement = Regex.Replace(result[column][i].ToString(), @"\t|\n|\r", "");
                        if (replacement.Length > rightAlign)
                        {
                            var newValue = replacement.Substring(0, rightAlign-3) + "...";
                            Console.Write(format, "|", newValue);
                        }
                        else
                        {
                            Console.Write(format, "|", replacement);
                        }
                    }

                    Console.WriteLine("|");
                }

                for (int i = 0; i < result.Keys.Count; i++)
                {
                    Console.Write("+");
                    for (int j = 0; j < leftAlign+rightAlign; j++)
                    {
                        Console.Write("-");
                    }
                }

                Console.WriteLine("+");
            }
    }
}