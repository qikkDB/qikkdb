using System;
using System.IO;
using ColmnarDB.NetworkClient;

namespace ColmnarDB.BenchmarkUtility
{
    public class UseDatabase
    {
        /// <summary>
        /// Method to use proper database
        /// </summary>
        /// <param name="database">name of chosen database</param>
        public void Use(string database, ColumnarDBClient client)
        {
            try
            {
                client.UseDatabase(database);
                Console.WriteLine(SuccessfulUse(database));
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
        /// Returns string for successful using chosen database
        /// </summary>
        /// <param name="database">name of database</param>
        /// <returns></returns>
        public static string SuccessfulUse(string database)
        {
            return string.Format("Successful using of database: '{0}'", database);
        }
        
        /// <summary>
        /// Return output when there is an unkown exception
        /// </summary>
        /// <returns>string about unknown exception prepared for console output</returns>
        public static string UnknownException()
        {
            return "Unknown exception has occured. Check if server is still running.";
        }
    }
}