using System;
using System.IO;
using ColmnarDB.NetworkClient;

namespace ColmnarDB.ConsoleClient
{
    /// <summary>
    /// Class to import CSV file
    /// </summary>
    public class ImportCSV
    {
        /// <summary>
        /// Method to import CSV file
        /// </summary>
        /// <param name="path">path to csv file that should be imported</param>
        /// <param name="database">database that is used</param>
        public void Import(string path, string database, ColumnarDBClient client)
        {
            try
            {
                throw new NotImplementedException("CSVImport semantics change, use integration platform, or run dropdbase_instarea.exe from console and use arguments in this form: 'dropdbase_instarea.exe csv_file_path database_name block_size' (if databaseName is not specified, default value 'TestDb' will be used; if blockSize is not specified, default value '1048576' will be used).");
                //client.ImportCSV(database, path);
                //Console.WriteLine(SuccessfulImport(path, database));
            }
            catch (FileNotFoundException)
            {
                Console.WriteLine(FileNotFound(path));
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
        /// Returns output for successful import of CSV file
        /// </summary>
        /// <param name="path">path to file</param>
        /// <returns>string about successful CSV import prepared for console output</returns>
        public static string SuccessfulImport(string path, string database)
        {
            return string.Format("Successful CSV import of file: '{0}', to '{1}' database", path, database);
        }

        /// <summary>
        /// Return output when imported CSV file is not found
        /// </summary>
        /// <param name="path">path to file</param>
        /// <returns>string about not finding CSV file</returns>
        public static string FileNotFound(string path)
        {
            return string.Format("Selected CSV file: '{0}' was not found. Check if the file exists.", path);
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