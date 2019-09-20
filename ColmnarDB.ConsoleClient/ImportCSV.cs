using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using ColmnarDB.NetworkClient;
using TellStory.Data;
using TellStory.Data.Parser;

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
        private long[] bytesImported;
        private long streamLength;

        /// <summary>
        /// Creates new instance of the ImportCSV object
        /// </summary>
        /// <param name="serverHostnameOrIP">IP or hostname of the database server</param>
        /// <param name="serverPort">Port of the database server</param>
        public ImportCSV(string ipAddress, short port)
        {
            this.ipAddress = ipAddress;
            this.port = port;
        }

        /// <summary>
        /// Method to import CSV file
        /// </summary>
        /// <param name="path">path to csv file that should be imported</param>
        /// <param name="database">database that is used</param>
        public void Import(string path, string database, bool hasHeader = true, char columnSeparator = '\0', int batchSize = 100000, int threadsCount = 0, Encoding encoding = null)
        {
            this.databaseName = database;            

            try
            {
                tableName = Path.GetFileNameWithoutExtension(path);
                
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
                linesImported = new long[threadsCount];
                bytesImported = new long[threadsCount];
                Thread[] threads = new Thread[threadsCount];

                // Each thread has its own instance of the Parser (because of different position of read) and ColumnarDBClient (because of concurrent bulk import)
                for (int i = 0; i < threadsCount; i++)
                {
                    long start = i * streamThreadLength;
                    long end = i * streamThreadLength + streamThreadLength + 1;
                    int index = i;
                    threads[index] = new Thread(() => this.ParseAndImportBatch(index, path, configuration, types, start, end));
                    threads[index].Start();
                    //this.ParseAndImportBatch(index, path, configuration, types, start, end);
                }
                
                for (int i = 0; i < threadsCount; i++)
                {
                    threads[i].Join();
                }

                for (int i = 0; i < threadsCount; i++)
                {
                    lines += linesImported[i];
                }

                var endTime = DateTime.Now;
                Console.WriteLine();
                Console.WriteLine("Importing done (" + (endTime - startTime).TotalSeconds.ToString() + "s.).");                

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

            var client = new ColumnarDBClient("Host=" + ipAddress + ";" + "Port=" + port.ToString() + ";");
            client.Connect();
            client.UseDatabase(databaseName);

            long lines = 0;
            while (true)
            {
                var outData = parserCSV.GetNextParsedDataBatch();
                if (outData == null)
                    break;
                lines += outData.GetCount();

                var colData = new NetworkClient.ColumnarDataTable(outData.GetColumnNames(), outData.GetColumnData(), outData.GetColumnTypes(), outData.GetColumnNames());
                colData.TableName = tableName;                
                
                client.BulkImport(colData);
                
                linesImported[threadId] = lines;
                bytesImported[threadId] = parserCSV.GetStreamPosition();
                
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
                Console.Write("\rImported " + totalLinesImported + " lines so far (" + Math.Round((float)totalBytesImported / streamLength * 100)  + "%)...");
            }

            client.Dispose();
        }
    }
}