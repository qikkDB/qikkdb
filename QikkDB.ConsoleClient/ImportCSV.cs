using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using ColmnarDB.NetworkClient;
using ColmnarDB.Types;

namespace ColmnarDB.ConsoleClient
{
    /// <summary>
    /// Class to import CSV file
    /// </summary>
    public class ImportCSV
    {
        private string ipAddress = "127.0.0.1";
        private short port = 12345;
        private string tableName;
        private string databaseName;
        private long[] linesImported;
        private long[] linesError;
        private long[] bytesImported;
        private long streamLength;
        private Mutex mutex = new Mutex();
        private ColumnarDBClient client;

        /// <summary>
        /// Creates new instance of the ImportCSV object
        /// </summary>
        /// <param name="serverHostnameOrIP">IP or hostname of the database server</param>
        /// <param name="serverPort">Port of the database server</param>
        public ImportCSV(string ipAddress, short port)
        {
            this.ipAddress = ipAddress;
            this.port = port;
            client = new ColumnarDBClient("Host=" + ipAddress + ";" + "Port=" + port.ToString() + ";");
        }

        /// <summary>
        /// Method to import CSV file
        /// </summary>
        /// <param name="path">Path to csv file that should be imported</param>
        /// <param name="databaseName">Database that is used</param>
        /// <param name="blockSize">Block size of table, if not specified default value of database server configuration will be used</param>
        /// <param name="hasHeader">Specifies whether input csv has header, if not specified default value is true</param>
        /// <param name="columnSeparator">Char representing column separator, if not specified default value will be guessed</param>
        /// <param name="batchSize">Number of lines processed in one batch, if not specified default value is 65536</param>
        /// <param name="threadsCount">Number of threads for processing csv, if not specified number of cores of the client CPU will be used</param>
        /// <param name="encoding">Encoding of csv, if not specified it will be guessed</param>
        public void Import(string path, string databaseName,
            string tableName = "", int blockSize = 0, bool hasHeader = true, char columnSeparator = '\0', int batchSize = 65536, int threadsCount = 0, Encoding encoding = null)
        {
            this.databaseName = databaseName;

            try
            {
                // Table name from path or from argument if present
                if (tableName == "")
                {
                    tableName = Path.GetFileNameWithoutExtension(path);
                }

                this.tableName = tableName;

                if (encoding == null)
                {
                    encoding = ParserCSV.GuessEncoding(path);
                }
                if (columnSeparator == '\0')
                {
                    columnSeparator = ParserCSV.GuessSeparator(path, encoding);
                }
                var types = ParserCSV.GuessTypes(path, hasHeader, columnSeparator, encoding);
                streamLength = ParserCSV.GetStreamLength(path);

                client.Connect();
                client.UseDatabase(databaseName);
                CreateTable(databaseName, tableName, types, blockSize);

                ParserCSV.Configuration configuration = new ParserCSV.Configuration(batchSize: batchSize, encoding: encoding, columnSeparator: columnSeparator);

                if (threadsCount <= 0)
                {
                    // Use more threads if file is bigger than 10MB
                    if (streamLength > 10000000)
                    {
                        threadsCount = Environment.ProcessorCount;
                    }
                    else
                    {
                        threadsCount = 1;
                    }
                }

                Console.WriteLine("Importing started with " + threadsCount + " threads.");
                var startTime = DateTime.Now;

                long streamThreadLength = streamLength / threadsCount;
                long lines = 0;
                long errors = 0;
                linesImported = new long[threadsCount];
                bytesImported = new long[threadsCount];
                linesError = new long[threadsCount];
                Thread[] threads = new Thread[threadsCount];
                Exception[] threadExceptions = new Exception[threadsCount];

                // Each thread has its own instance of the Parser (because of different position of read) and ColumnarDBClient (because of concurrent bulk import)
                for (int i = 0; i < threadsCount; i++)
                {
                    long start = i * streamThreadLength;
                    long end = i * streamThreadLength + streamThreadLength + 1;
                    int index = i;
                    threadExceptions[index] = null;
                    threads[index] = new Thread(() =>
                    {
                        try
                        {
                            this.ParseAndImportBatch(index, path, configuration, types, start, end);
                        }
                        catch (Exception ex)
                        {
                            threadExceptions[index] = ex;
                        }
                    });
                    threads[index].Start();
                    //this.ParseAndImportBatch(index, path, configuration, types, start, end);
                }

                for (int i = 0; i < threadsCount; i++)
                {
                    threads[i].Join();
                }

                for (int i = 0; i < threadsCount; i++)
                {
                    if (threadExceptions[i] != null)
                    {
                        throw threadExceptions[i];
                    }
                }

                for (int i = 0; i < threadsCount; i++)
                {
                    lines += linesImported[i];
                    errors += linesError[i];
                }

                client.Dispose();

                var endTime = DateTime.Now;
                Console.WriteLine();
                Console.WriteLine("Importing done (imported " + lines + " records, " + errors + " failed, " + (endTime - startTime).TotalSeconds.ToString() + "sec.).");

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
            catch (ParserCSV.ParserException e)
            {
                Console.WriteLine("Parser Exception: " + e.Message);
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

        /// <summary>
        /// Parses stream in batches and each batch is imported by the client
        /// </summary>
        /// <param name="threadId">Id of thread starting from 0</param>
        /// <param name="file">Name of the file with full path</param>
        /// <param name="configuration">Configuration for parser with specified parameters (column separator, encoding, etc.)</param>
        /// <param name="types">Dictionary describing imported table with tuples (column name, column type)</param>
        /// <param name="startBytePosition">Start reading position in file</param>
        /// <param name="endBytePosition">End reading position in file</param>
        /// <returns>string about not finding CSV file</returns>
        private void ParseAndImportBatch(int threadId, string file, ParserCSV.Configuration configuration, Dictionary<string, Type> types, long startBytePosition = 0, long endBytePosition = 0)
        {
            var stream = File.Open(file, FileMode.Open, FileAccess.Read, FileShare.Read);
            var streamReader = new StreamReader(stream, configuration.Encoding);
            ParserCSV parserCSV = new ParserCSV(streamReader: streamReader, tableName: tableName, configuration: configuration, types: types, startBytePosition: startBytePosition, endBytePosition: endBytePosition);

            long lines = 0;
            long errors = 0;
            while (true)
            {
                long batchImportedLines;
                long batchErrorLines;

                var outData = parserCSV.GetNextParsedDataBatch(out batchImportedLines, out batchErrorLines);

                if (outData == null)
                    break;
                lines += batchImportedLines;
                errors += batchErrorLines;

                var colData = new NetworkClient.ColumnarDataTable(outData.GetColumnNames(), outData.GetColumnData(), outData.GetColumnTypes(), outData.GetColumnNames());
                colData.TableName = tableName;

                mutex.WaitOne();
                try
                {
                    client.BulkImport(colData);
                }
                catch (Exception)
                {
                    mutex.ReleaseMutex();
                    throw;
                }
                mutex.ReleaseMutex();

                linesImported[threadId] = lines;
                bytesImported[threadId] = parserCSV.GetStreamPosition();
                linesError[threadId] = errors;

                long totalLinesImported = 0;
                for (int i = 0; i < linesImported.Length; i++)
                {
                    totalLinesImported += linesImported[i];
                }

                long totalBytesImported = 0;
                for (int i = 0; i < bytesImported.Length; i++)
                {
                    totalBytesImported += bytesImported[i];
                }

                long totalLinesError = 0;
                for (int i = 0; i < linesError.Length; i++)
                {
                    totalLinesError += linesError[i];
                }
                Console.Write("\rImported " + totalLinesImported + " records so far (" + Math.Min(Math.Round((float)totalBytesImported / streamLength * 100), 100) + "%)...");
            }

            if (stream != null)
            {
                stream.Dispose();
            }
        }


        /// <summary>
        /// Method to create table in database
        /// </summary>
        /// <param name="databaseName">Database in which table will be created</param>
        /// <param name="tableName">Table name</param>
        /// <param name="types">Dictionary describing imported table with tuples (column name, column type)</param>
        /// <param name="blockSize">Block size of table, if not specified default value is used</param>
        private void CreateTable(string databaseName, string tableName, Dictionary<string, Type> types, int blockSize = 0)
        {
            // build query for creating table
            string query = "CREATE TABLE ";

            query += "[" + tableName + "] ";
            if (blockSize > 0)
            {
                query += "" + blockSize.ToString() + " ";
            }
            query += "(";
            int i = 0;
            foreach (var typePair in types)
            {
                query += "[" + typePair.Key + "] " + GetDatabaseTypeName(typePair.Value);
                if (++i < types.Count)
                {
                    query += ", ";
                }
            }
            query += ");";

            // check if database exists
            try
            {
                try
                {
                    client.Query(query);
                }
                catch (QueryException)
                {
                    // this exception does not show message and it is necessary to get next result set
                }
                catch (Exception)
                {
                }

                // get next result to see if new error
                client.GetNextQueryResult();
            }
            catch (QueryException e)
            {
                if (e.Message.Contains("exists"))
                {
                    Console.WriteLine("Table " + tableName + " already exits, records will be appended.");
                }
                else
                {
                    Console.WriteLine(UnknownException() + e.Message);
                }

            }
            catch (Exception e)
            {
                Console.WriteLine(UnknownException() + e.Message);
            }

        }

        /// <summary>
        /// Method to convert .Net type to name of database type
        /// </summary>
        /// <param name="type">Type of .NET</param>
        /// <returns>Name of database type</returns>
        private string GetDatabaseTypeName(Type type)
        {
            string result = "STRING";

            if (type == typeof(Int32))
                result = "INT";
            else if (type == typeof(Int64))
                result = "LONG";
            else if (type == typeof(float))
                result = "FLOAT";
            else if (type == typeof(double))
                result = "DOUBLE";
            else if (type == typeof(DateTime))
                result = "LONG";
            else if (type == typeof(bool))
                result = "BOOL";
            else if (type == typeof(Point))
                result = "GEO_POINT";
            else if (type == typeof(ComplexPolygon))
                result = "GEO_POLYGON";
            else if (type == typeof(string))
                result = "STRING";

            return result;
        }
    }
}